import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from core.dataset import PairedPatchDataset
from core.logging.logger import DistributedSimpleLogger2
from core.training import AMPGradScaler, count_model_parameters, set_random_seed
from flow_matching.path import CondOTProbPath
from models.wrapper import (
    DIT_TRANSFORMER_2D_CONFIGS,
    DiTTransformer2DWrapper,
    build_dit_transformer_2d_wrapper,
)
from training import distributed_mode


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a conditional flow-matching model from noise to seismic patches with dimension coordinates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_data_dir",
        default="./dataset/train",
        help="Directory containing training image patch NPY files.",
    )
    parser.add_argument(
        "--train_data_dim_dir",
        default="./dataset/dim_train",
        help="Directory containing training dimension-coordinate patch NPY files.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Directory used for logs and checkpoints.",
    )
    parser.add_argument(
        "--model_arch",
        choices=sorted(DIT_TRANSFORMER_2D_CONFIGS.keys()),
        default="DiT_T_4",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--input_size",
        default=64,
        type=int,
        help="Height and width of the square training patches.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint directory saved by a previous train6.py run.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
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
        "--clip_grad",
        default=1.0,
        type=float,
        help="Max gradient norm. Set <= 0 to disable gradient clipping.",
    )
    parser.add_argument(
        "--upcast_attention",
        action="store_true",
        help="Run attention score computation in fp32 for better mixed-precision stability.",
    )
    parser.add_argument(
        "--num_epochs",
        default=1000,
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
        default=50,
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
    dataset = PairedPatchDataset(args.train_data_dir, args.train_data_dim_dir)

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

    return dataset, dataloader


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        scaler,
        flow_path,
        device,
        args,
        logger,
        epoch,
):
    gc.collect()
    model.train(True)

    running_loss = 0.0
    running_steps = 0
    epoch_loss = 0.0
    epoch_steps = 0
    total_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        if step % args.grad_accum_steps == 0:
            optimizer.zero_grad()
            running_loss = 0.0
            running_steps = 0

        clean_images, conditioning = batch
        clean_images = clean_images.to(device, non_blocking=True)
        conditioning = conditioning.to(device, non_blocking=True)
        if clean_images.shape[1] != 1:
            raise ValueError(
                f"train6.py expects single-channel image patches, got shape {tuple(clean_images.shape)}."
            )
        if clean_images.shape[-2:] != (args.input_size, args.input_size):
            raise ValueError(
                f"Expected {args.input_size}x{args.input_size} patches, "
                f"got shape {tuple(clean_images.shape)}."
            )
        if conditioning.shape[-2:] != (args.input_size, args.input_size):
            raise ValueError(
                f"Expected {args.input_size}x{args.input_size} dimension patches, "
                f"got shape {tuple(conditioning.shape)}."
            )

        noise = torch.randn_like(clean_images)
        timesteps = torch.rand(clean_images.shape[0], device=device)
        flow_sample = flow_path.sample(t=timesteps, x_0=noise, x_1=clean_images)
        noisy_images = flow_sample.x_t
        target_velocity = flow_sample.dx_t

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predicted_velocity = model(
                noisy_images,
                timesteps,
                extra={"concat_conditioning": conditioning},
            )
            loss = F.mse_loss(predicted_velocity, target_velocity)

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        running_steps += 1
        epoch_loss += loss_value
        epoch_steps += 1

        scaled_loss = loss / args.grad_accum_steps
        should_step = (step + 1) % args.grad_accum_steps == 0
        step_start_time = time.time()
        clip_grad = args.clip_grad if args.clip_grad > 0 else None
        grad_norm = scaler(
            scaled_loss,
            optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
            update_grad=should_step,
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        train_fields = {
            "epoch": epoch + 1,
            "step": step + 1,
            "total_steps": total_steps,
            "batch_size": int(clean_images.shape[0]),
            "loss": loss_value,
            "running_loss": running_loss / max(running_steps, 1),
            "lr": learning_rate,
            "optimizer_step": int(should_step),
            "grad_norm": "" if grad_norm is None else float(grad_norm.detach().cpu()),
            "clip_grad": "" if clip_grad is None else clip_grad,
            "step_time_sec": time.time() - step_start_time,
        }
        logger.log_train(**train_fields)

    return epoch_loss / max(epoch_steps, 1)


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
    logger.log_system_info(
        package_names=[
            "torch",
            "torchvision",
            "numpy",
            "flow_matching",
        ]
    )
    logger.log_info_block(
        "GLOBAL PARAMETERS",
        {
            "task": "dim_conditioned_flow_matching_seismic_generation",
            "train_data_dir": args.train_data_dir,
            "train_data_dim_dir": args.train_data_dim_dir,
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "model_arch": args.model_arch,
            "model_config": dict(model.model.config),
            "dim_channels": dataset.dataset1[0].shape[0],
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
            "amp": True,
            "model": str(model),
        }
    )


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

    logger.log_event(
        "dataset_initializing",
        train_data_dir=args.train_data_dir,
        train_data_dim_dir=args.train_data_dim_dir,
    )
    dataset, train_loader = build_dataloader(args)
    dim_channels = dataset.dataset1[0].shape[0]
    logger.log_event(
        "dataset_initialized",
        dataset_size=len(dataset),
        dim_channels=int(dim_channels),
        num_batches=len(train_loader),
    )

    logger.log_event("model_initializing", model_arch=args.model_arch)
    model = build_dit_transformer_2d_wrapper(
        model_arch=args.model_arch,
        in_channels=1 + dim_channels,
        out_channels=1,
        sample_size=args.input_size,
        num_embeds_ada_norm=1,
        upcast_attention=args.upcast_attention,
        device=device,
    )
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
    flow_path = CondOTProbPath()

    start_epoch = 0
    if args.ckpt:
        logger.log_event("checkpoint_loading", path=args.ckpt)
        loaded_model, start_epoch, training_state = (
            DiTTransformer2DWrapper.from_training(
                save_directory=args.ckpt,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                device=device,
            )
        )
        model_without_ddp.model.load_state_dict(loaded_model.model.state_dict())
        logger.log_event(
            "checkpoint_loaded",
            path=args.ckpt,
            checkpoint_epoch=start_epoch,
            start_epoch=start_epoch + 1,
        )

    logger.log_event("training_started")
    start_time = time.time()
    checkpoint_dir = logger.run_dir

    for epoch in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            flow_path=flow_path,
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
            checkpoint_path = (
                    Path(checkpoint_dir)
                    / f"checkpoint_epoch_{epoch + 1:05d}"
            )
            model_without_ddp.save_training(
                save_directory=checkpoint_path,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch + 1,
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
    logger.close()

    if args.distributed:
        distributed_mode.barrier([args.gpu])
        distributed_mode.destroy()


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
