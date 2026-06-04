import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from core.dataset import PatchDataset
from core.logging.logger import DistributedSimpleLogger2
from core.training import AMPGradScaler, count_model_parameters, set_random_seed
from models.seismic_vae import SeismicSpatialVAE, kl_divergence
from training import distributed_mode


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a seismic spatial VAE on single-channel seismic patch NPY files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_data_dir",
        default="./train_dataset",
        help="Directory containing training patch NPY files normalized to [-1, 1].",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Directory used for logs and checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Mini-batch size per process.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        default=1,
        type=int,
        help="Number of mini-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "linear"],
        default="constant",
        help="Learning-rate schedule to use during training.",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of DataLoader worker processes per training process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pinned host memory in the DataLoader.",
    )
    parser.add_argument(
        "--save_every_epochs",
        default=10,
        type=int,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--log_id",
        default=None,
        help="Optional run directory name under output_dir.",
    )
    parser.add_argument(
        "--log_console",
        action="store_true",
        help="Also print SimpleLogger2 output to stdout.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Base random seed. Distributed ranks add their rank to this value.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device, such as cuda, cuda:0, cpu, or mps.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="First beta coefficient for AdamW.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Second beta coefficient for AdamW.",
    )
    parser.add_argument(
        "--input_size",
        default=256,
        choices=[128, 256, 512],
        type=int,
        help="Input patch height and width.",
    )
    parser.add_argument(
        "--input_channels",
        default=1,
        type=int,
        help="Number of input patch channels.",
    )
    parser.add_argument(
        "--latent_channels",
        default=4,
        type=int,
        help="Latent channel count for [B,C,latent_size,latent_size].",
    )
    parser.add_argument(
        "--latent_size",
        default=32,
        choices=[16, 32, 64],
        type=int,
        help="Spatial latent size for the CxNxN latent map.",
    )
    parser.add_argument(
        "--hidden_channels",
        default=32,
        type=int,
        help="Base hidden channel count for the VAE encoder and decoder.",
    )
    parser.add_argument(
        "--channel_multipliers",
        nargs="*",
        type=int,
        default=None,
        help="Optional encoder/decoder channel multipliers. Defaults to powers of two.",
    )
    parser.add_argument(
        "--autoencoder",
        action="store_true",
        help="Disable latent sampling and use z=mu.",
    )
    parser.add_argument(
        "--kl_weight",
        default=0.0,
        type=float,
        help="Weight for optional KL regularization. Set 0 to disable.",
    )
    parser.add_argument(
        "--dist_on_itp",
        action="store_true",
        help="Initialize distributed training from ITP/OpenMPI environment variables.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="URL used to initialize distributed training.",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of distributed processes, usually provided by torchrun.",
    )
    return parser


def build_dataloader(args):
    dataset = PatchDataset(args.train_data_dir)

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=distributed_mode.get_world_size(),
        rank=distributed_mode.get_rank(),
        shuffle=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    return dataset, sampler, dataloader


def build_model(args, device):
    model = SeismicSpatialVAE(
        input_channels=args.input_channels,
        output_channels=1,
        latent_channels=args.latent_channels,
        input_size=args.input_size,
        latent_size=args.latent_size,
        hidden_channels=args.hidden_channels,
        channel_multipliers=tuple(args.channel_multipliers)
        if args.channel_multipliers
        else None,
        use_vae=not args.autoencoder,
    ).to(device)
    return model


def compute_loss(outputs, clean_images, args):
    recon = outputs["recon"]
    mse = F.mse_loss(recon, clean_images)
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    total = mse + args.kl_weight * kl
    return {
        "mse": mse,
        "kl": kl,
        "total": total,
    }


def training_mode_name(args):
    if args.autoencoder and args.kl_weight > 0:
        return "autoencoder_kl_regularized"
    if args.autoencoder:
        return "autoencoder"
    if args.kl_weight > 0:
        return "weak_kl_vae"
    return "sampled_vae_no_kl"


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        scaler,
        device,
        args,
        logger,
        epoch,
):
    gc.collect()
    model.train(True)

    running_total_loss = 0.0
    running_mse_loss = 0.0
    running_kl_loss = 0.0
    running_steps = 0
    epoch_total_loss = 0.0
    epoch_steps = 0
    total_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        if step % args.grad_accum_steps == 0:
            optimizer.zero_grad()
            running_total_loss = 0.0
            running_mse_loss = 0.0
            running_kl_loss = 0.0
            running_steps = 0

        clean_images = batch.to(device, non_blocking=True)
        if clean_images.shape[1] != args.input_channels:
            raise ValueError(
                f"train_vae.py expected {args.input_channels} input channels, got {clean_images.shape[1]}."
            )
        if clean_images.shape[-2:] != (args.input_size, args.input_size):
            raise ValueError(
                f"train_vae.py expected patches with shape {args.input_size}x{args.input_size}, "
                f"got {tuple(clean_images.shape[-2:])}."
            )

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(clean_images)
            losses = compute_loss(outputs, clean_images, args)

        total_loss_value = float(losses["total"].detach().cpu())
        mse_loss_value = float(losses["mse"].detach().cpu())
        kl_loss_value = float(losses["kl"].detach().cpu())
        running_total_loss += total_loss_value
        running_mse_loss += mse_loss_value
        running_kl_loss += kl_loss_value
        running_steps += 1
        epoch_total_loss += total_loss_value
        epoch_steps += 1

        loss = losses["total"] / args.grad_accum_steps
        should_step = (step + 1) % args.grad_accum_steps == 0
        scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=should_step,
        )

        if should_step:
            optimizer.zero_grad()

        logger.log_train(
            epoch=epoch + 1,
            step=step + 1,
            total_steps=total_steps,
            batch_size=int(clean_images.shape[0]),
            loss=total_loss_value,
            running_loss=running_total_loss / max(running_steps, 1),
            mse_loss=mse_loss_value,
            running_mse_loss=running_mse_loss / max(running_steps, 1),
            kl_loss=kl_loss_value,
            running_kl_loss=running_kl_loss / max(running_steps, 1),
            kl_weight=args.kl_weight,
            lr=optimizer.param_groups[0]["lr"],
            optimizer_step_taken=int(should_step),
        )

    return epoch_total_loss / max(epoch_steps, 1)


def log_training_info(
        logger,
        args,
        dataset,
        dataloader,
        model,
        total_params,
        trainable_params,
        frozen_params,
        effective_batch_size,
):
    if logger is None:
        return
    logger.log_system_info()
    logger.log_info_block(
        "GLOBAL PARAMETERS",
        {
            "task": "seismic_spatial_vae",
            "train_data_dir": args.train_data_dir,
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "normalization": "pre_normalized_minus_one_to_one",
            "input_channels": args.input_channels,
            "output_channels": 1,
            "input_size": args.input_size,
            "latent_channels": args.latent_channels,
            "latent_size": args.latent_size,
            "latent_shape": [args.latent_channels, args.latent_size, args.latent_size],
            "hidden_channels": args.hidden_channels,
            "channel_multipliers": args.channel_multipliers,
            "mode": training_mode_name(args),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": effective_batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "lr_schedule": args.lr_schedule,
            "loss": "mse_plus_kl",
            "kl_weight": args.kl_weight,
            "amp": True,
            "model": str(model),
        },
    )


def save_training_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        scaler,
        args,
        epoch,
        output_dir,
):
    checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch + 1:05d}.pth"
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "model_config": model.config.__dict__,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def main(args):
    distributed_mode.init_distributed_mode(args)

    logger = DistributedSimpleLogger2(
        output_dir=args.output_dir,
        log_id=args.log_id,
        distributed=args.distributed,
        rank=getattr(args, "rank", 0),
        world_size=getattr(args, "world_size", 1),
        local_rank=getattr(args, "gpu", 0),
        overwrite=True,
        console=args.log_console and distributed_mode.get_rank() == 0,
    )
    args.log_id = logger.log_id
    logger.log_event(
        "script_started",
        job_dir=os.path.dirname(os.path.realpath(__file__)),
        log_file=logger.log_file,
    )
    logger.log_node_info()
    logger.log_info_block("ARGPARSE PARAMETERS", args)

    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    set_random_seed(seed)

    logger.log_event("dataset_initializing", train_data_dir=args.train_data_dir)
    dataset, train_sampler, train_loader = build_dataloader(args)
    logger.log_event(
        "dataset_initialized",
        dataset_size=len(dataset),
        num_batches=len(train_loader),
        sampler=str(train_sampler),
    )

    logger.log_event("model_initializing")
    model = build_model(args, device)
    model_without_ddp = model

    total_params, trainable_params, frozen_params = count_model_parameters(model_without_ddp)

    effective_batch_size = (
            args.batch_size * args.grad_accum_steps * distributed_mode.get_world_size()
    )
    log_training_info(
        logger,
        args=args,
        dataset=dataset,
        dataloader=train_loader,
        model=model_without_ddp,
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        effective_batch_size=effective_batch_size,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    logger.log_event("optimizer_initializing")
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    if args.lr_schedule == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.num_epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.learning_rate,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            total_iters=args.num_epochs,
            factor=1.0,
        )

    logger.log_event(
        "optimizer_initialized",
        optimizer=str(optimizer),
        lr_scheduler=str(lr_scheduler),
    )

    scaler = AMPGradScaler(enabled=device.type == "cuda", device=device.type)

    logger.log_event("training_started")
    start_time = time.time()
    checkpoint_dir = logger.run_dir

    for epoch in range(args.num_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            logger=logger,
            epoch=epoch,
        )
        lr_scheduler.step()
        logger.log_event("epoch_finished", epoch=epoch + 1, mean_loss=epoch_loss, )

        if (
                (epoch + 1) % args.save_every_epochs == 0
                and distributed_mode.get_rank() == 0
        ):
            checkpoint_path = save_training_checkpoint(
                model=model_without_ddp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                output_dir=checkpoint_dir,
            )
            logger.log_event(
                "checkpoint_saved",
                epoch=epoch + 1,
                path=str(checkpoint_path),
            )

    total_time = time.time() - start_time
    logger.log_event(
        "training_finished",
        total_time_sec=total_time,
        run_dir=str(checkpoint_dir),
    )

    if args.distributed:
        logger.log_event("final_barrier_started", rank=args.rank, gpu=args.gpu)
        distributed_mode.barrier([args.gpu])
        logger.log_event("final_barrier_finished", rank=args.rank, gpu=args.gpu)
        logger.log_event("distributed_destroy_started", rank=args.rank)
        distributed_mode.destroy()
        logger.log_event("distributed_destroy_finished", rank=args.rank)

    logger.close()


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
