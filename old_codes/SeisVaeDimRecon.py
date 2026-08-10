import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.dataset import PairedPatchDataset, PatchDataset
from core.logging.logger import build_dist_logger
from core.training import AMPGradScaler, count_model_parameters, set_random_seed
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.wrapper import (
    AutoencoderKLWrapper,
    DIT_TRANSFORMER_2D_CONFIGS,
    DiTTransformer2DWrapper,
    build_dit_transformer_2d_wrapper,
)
from training import distributed_mode


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a dimension-conditioned flow-matching model in VAE latent space."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  python SeisVaeDimRecon.py "
            "--ckpt_vae ./output_seis_vae/run/checkpoint_epoch_00100 "
            "--train_data_dir ./dataset256/train "
            "--train_data_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_seis_vae_dim_recon "
            "--model_arch DiT_T_4 --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Valid:\n"
            "  python SeisVaeDimRecon.py valid "
            "--ckpt ./output_seis_vae_dim_recon/run/checkpoint_epoch_00100 "
            "--ckpt_vae ./output_seis_vae/run/checkpoint_epoch_00100 "
            "--train_data_dim_dir ./dataset256/valid_dim "
            "--output_dir ./output_seis_vae_dim_recon/valid "
            "--batch_size 32 --device cuda\n"
        ),
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "valid"],
        default="train",
        help="Run mode. Omit for training.",
    )
    parser.add_argument(
        "--train_data_dir",
        default="./dataset/train",
        help="Directory containing training image patch NPY files.",
    )
    parser.add_argument(
        "--train_data_dim_dir",
        default="./dataset/train_dim",
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
        default=256,
        type=int,
        help="Deprecated. DiT sample size is inferred from the VAE latent shape.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint directory saved by a previous SeisVaeDimRecon.py run.",
    )
    parser.add_argument(
        "--ckpt_vae",
        default=None,
        help="VAE checkpoint directory saved by SeisVae.py/train_seismic_vae.py.",
    )
    parser.add_argument(
        "--solver_step_size",
        default=0.05,
        type=float,
        help="Euler solver step size used in valid mode.",
    )
    parser.add_argument(
        "--clip_recon",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Clip reconstructed valid output data to [MIN, MAX]. Disabled by default.",
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


def validate_train_args(args):
    validate_common_args(args, mode="train")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("--num_epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive.")
    if args.save_every_epochs <= 0:
        raise ValueError("--save_every_epochs must be positive.")


def validate_valid_args(args):
    validate_common_args(args, mode="valid")
    if args.ckpt is None:
        raise ValueError("--ckpt is required in valid mode.")
    if not Path(args.ckpt).is_dir():
        raise FileNotFoundError(f"--ckpt must be a SeisVaeDimRecon.py checkpoint directory, got {args.ckpt}.")
    if not Path(args.train_data_dim_dir).is_dir():
        raise FileNotFoundError(f"--train_data_dim_dir must be a directory, got {args.train_data_dim_dir}.")
    if args.solver_step_size <= 0:
        raise ValueError("--solver_step_size must be positive.")
    if args.clip_recon is not None and args.clip_recon[0] >= args.clip_recon[1]:
        raise ValueError("--clip_recon MIN must be smaller than MAX.")


def validate_common_args(args, mode):
    if args.ckpt_vae is None:
        raise ValueError(f"--ckpt_vae is required in {mode} mode.")
    if not Path(args.ckpt_vae).is_dir():
        raise FileNotFoundError(f"--ckpt_vae must be a VAE checkpoint directory, got {args.ckpt_vae}.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")


def load_vae_model(args, device, logger):
    checkpoint_dir = Path(args.ckpt_vae)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"--ckpt_vae must be a VAE checkpoint directory, got {checkpoint_dir}.")

    logger.log_event("vae_loading", checkpoint=str(checkpoint_dir))
    vae = AutoencoderKLWrapper.from_pretrained(checkpoint_dir, device=device)
    vae.eval()
    vae.requires_grad_(False)
    logger.log_event(
        "vae_loaded",
        checkpoint=str(checkpoint_dir),
        input_channels=int(vae.model.config.in_channels),
        latent_channels=int(vae.model.config.latent_channels),
        sample_size=vae.model.config.sample_size,
        scaling_factor=vae.model.config.scaling_factor,
    )
    return vae


def get_vae_sample_size(vae):
    sample_size = vae.model.config.sample_size
    if isinstance(sample_size, (list, tuple)):
        return tuple(int(value) for value in sample_size)
    size = int(sample_size)
    return size, size


def validate_vae_input_images(vae, images):
    expected_input_channels = int(vae.model.config.in_channels)
    if images.shape[1] != expected_input_channels:
        raise ValueError(
            f"VAE expects {expected_input_channels} input channels, got shape {tuple(images.shape)}."
        )
    expected_image_size = get_vae_sample_size(vae)
    if images.shape[-2:] != expected_image_size:
        raise ValueError(
            f"VAE expects image patches with shape {expected_image_size}, "
            f"got shape {tuple(images.shape)}."
        )


def encode_vae_latents(vae, images):
    posterior = vae.model.encode(images).latent_dist
    return posterior.mode()


def infer_vae_latent_shape(vae, dataset, device):
    sample = dataset.dataset0[0].unsqueeze(0).to(device)
    with torch.inference_mode():
        latents = encode_vae_latents(vae, sample)
    return tuple(int(value) for value in latents.shape[1:])


def infer_vae_latent_shape_from_config(vae, device):
    height, width = get_vae_sample_size(vae)
    channels = int(vae.model.config.in_channels)
    sample = torch.zeros((1, channels, height, width), device=device)
    with torch.inference_mode():
        latents = encode_vae_latents(vae, sample)
    return tuple(int(value) for value in latents.shape[1:])


def validate_square_latent_shape(vae_latent_shape):
    _, latent_height, latent_width = vae_latent_shape
    if latent_height != latent_width:
        raise ValueError(f"Expected square VAE latents, got shape {vae_latent_shape}.")


def prepare_vae_and_latent_shape(args, device, logger, dataset=None):
    vae = load_vae_model(args, device, logger)
    if dataset is None:
        vae_latent_shape = infer_vae_latent_shape_from_config(vae, device)
    else:
        vae_latent_shape = infer_vae_latent_shape(vae, dataset, device)
    validate_square_latent_shape(vae_latent_shape)
    return vae, vae_latent_shape


def resize_conditioning_to_latent(conditioning, latent_spatial_size):
    return F.adaptive_avg_pool2d(conditioning, output_size=latent_spatial_size)


def validate_dim_latent_model_channels(model, latent_channels, dim_channels):
    expected_model_in_channels = latent_channels + dim_channels
    model_in_channels = int(model.model.config.in_channels)
    model_out_channels = int(model.model.config.out_channels)
    if model_in_channels != expected_model_in_channels:
        raise ValueError(
            "Checkpoint input channel count does not match VAE latent + dim data: "
            f"model in_channels={model_in_channels}, "
            f"latent_channels={latent_channels}, dim_channels={dim_channels}, "
            f"expected={expected_model_in_channels}."
        )
    if model_out_channels != latent_channels:
        raise ValueError(
            "Checkpoint output channel count does not match VAE latent channels: "
            f"model out_channels={model_out_channels}, latent_channels={latent_channels}."
        )


def split_patch_files_for_rank(dataset):
    rank = distributed_mode.get_rank()
    world_size = distributed_mode.get_world_size()
    return dataset.patch_files[rank::world_size]


def make_conditioning_batch(array, start, end, device):
    batch = np.array(array[start:end], copy=True)
    if batch.ndim == 3:
        batch = batch[:, np.newaxis, :, :]
    tensor = torch.from_numpy(batch).float()
    return tensor.to(device, non_blocking=True)


def get_reconstruction_output_file(input_file, input_root, output_root):
    relative_path = Path(input_file).relative_to(input_root)
    return Path(output_root) / relative_path


def restore_reconstruction_file_shape(reconstructed, input_array):
    if reconstructed.shape[1] == 1 and input_array.ndim == 3:
        return reconstructed.squeeze(1).numpy()
    return reconstructed.numpy()


def reconstruct_dim_file(
        input_file,
        input_root,
        output_root,
        vae,
        solver,
        time_grid,
        device,
        args,
        logger,
        latent_channels,
        latent_height,
        latent_width,
        file_index,
):
    dim_array = np.load(input_file, mmap_mode="r")
    num_patches = int(dim_array.shape[0])
    output_file = get_reconstruction_output_file(input_file, input_root, output_root)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reconstructed_batches = []
    file_min = None
    file_max = None
    rank = distributed_mode.get_rank()

    for batch_start in range(0, num_patches, args.batch_size):
        batch_end = min(batch_start + args.batch_size, num_patches)
        conditioning = make_conditioning_batch(dim_array, batch_start, batch_end, device)
        conditioning = resize_conditioning_to_latent(
            conditioning,
            (latent_height, latent_width),
        )
        latent_noise = torch.randn(
            (
                conditioning.shape[0],
                latent_channels,
                latent_height,
                latent_width,
            ),
            device=device,
            dtype=conditioning.dtype,
        )
        sampled_latents = solver.sample(
            time_grid=time_grid,
            x_init=latent_noise,
            return_intermediates=False,
            step_size=args.solver_step_size,
            cfg_scale=0.0,
            label=None,
            concat_conditioning={"concat_conditioning": conditioning},
        )
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            reconstructed = vae.model.decode(sampled_latents).sample
        reconstructed = reconstructed.detach().float().cpu()
        if args.clip_recon is not None:
            reconstructed = reconstructed.clamp(
                min=float(args.clip_recon[0]),
                max=float(args.clip_recon[1]),
            )
        reconstructed_batches.append(reconstructed)
        batch_min = float(reconstructed.min())
        batch_max = float(reconstructed.max())
        file_min = batch_min if file_min is None else min(file_min, batch_min)
        file_max = batch_max if file_max is None else max(file_max, batch_max)

        logger.log_valid(
            rank=rank,
            file_index=file_index,
            file_name=Path(input_file).name,
            batch_start=batch_start,
            batch_end=batch_end,
            batch_size=batch_end - batch_start,
        )

    reconstructed_patches = torch.cat(reconstructed_batches, dim=0)
    reconstructed_array = restore_reconstruction_file_shape(reconstructed_patches, dim_array)
    np.save(output_file, reconstructed_array)

    return {
        "rank": rank,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "num_patches": num_patches,
        "output_shape": list(reconstructed_array.shape),
        "output_min": file_min,
        "output_max": file_max,
    }


def gather_validation_summaries(local_summary):
    if not distributed_mode.is_dist_avail_and_initialized():
        return [local_summary]
    summaries = [None for _ in range(distributed_mode.get_world_size())]
    torch.distributed.all_gather_object(summaries, local_summary)
    return summaries


class DimLatentVelocityModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(
            self,
            x,
            t,
            cfg_scale,
            label,
            concat_conditioning,
    ):
        del cfg_scale, label

        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])

        with torch.inference_mode():
            with torch.amp.autocast(device_type=x.device.type, enabled=x.device.type == "cuda"):
                result = self.model(x, t, extra=concat_conditioning)
        return result.to(dtype=torch.float32)


def train_one_epoch(
        model,
        vae,
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
        validate_vae_input_images(vae, clean_images)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                clean_latents = encode_vae_latents(vae, clean_images)
        conditioning = resize_conditioning_to_latent(conditioning, clean_latents.shape[-2:])

        noise = torch.randn_like(clean_latents)
        timesteps = torch.rand(clean_latents.shape[0], device=device)
        flow_sample = flow_path.sample(t=timesteps, x_0=noise, x_1=clean_latents)
        noisy_latents = flow_sample.x_t
        target_velocity = flow_sample.dx_t

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predicted_velocity = model(
                noisy_latents,
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
            "batch_size": int(clean_latents.shape[0]),
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
        vae_latent_shape,
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
            "ckpt_vae": args.ckpt_vae,
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "model_arch": args.model_arch,
            "model_config": dict(model.model.config),
            "dim_channels": dataset.dataset1[0].shape[0],
            "vae_latent_shape": list(vae_latent_shape),
            "conditioning_resize": "adaptive_avg_pool2d_to_vae_latent_spatial_size",
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


def run_train(args):
    validate_train_args(args)
    distributed_mode.init_distributed_mode(args)

    logger = build_dist_logger(args, log_node_info=True)

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
    vae, vae_latent_shape = prepare_vae_and_latent_shape(args, device, logger, dataset=dataset)
    latent_channels, latent_height, latent_width = vae_latent_shape
    logger.log_event(
        "dataset_initialized",
        dataset_size=len(dataset),
        dim_channels=int(dim_channels),
        num_batches=len(train_loader),
        vae_latent_shape=list(vae_latent_shape),
    )

    logger.log_event("model_initializing", model_arch=args.model_arch)
    model = build_dit_transformer_2d_wrapper(
        model_arch=args.model_arch,
        in_channels=latent_channels + dim_channels,
        out_channels=latent_channels,
        sample_size=latent_height,
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
        vae_latent_shape=vae_latent_shape,
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
            DiTTransformer2DWrapper.from_pretrained(
                save_directory=args.ckpt,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                device=device,
                return_training_state=True,
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
            vae=vae,
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
            model_without_ddp.save_pretrained(
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


def run_valid(args):
    validate_valid_args(args)
    distributed_mode.init_distributed_mode(args)

    logger = build_dist_logger(
        args,
        log_id=args.log_id or ".",
        distributed=getattr(args, "distributed", False),
        rank=distributed_mode.get_rank(),
        world_size=distributed_mode.get_world_size(),
        local_rank=getattr(args, "gpu", 0),
        console=args.log_console,
        logs=[
            "rank",
            "file_index",
            "file_name",
            "batch_start",
            "batch_end",
            "batch_size",
        ],
    )
    output_dir = logger.run_dir
    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    set_random_seed(seed)
    np.random.seed(seed)

    vae, vae_latent_shape = prepare_vae_and_latent_shape(args, device, logger)
    latent_channels, latent_height, latent_width = vae_latent_shape

    logger.log_event("checkpoint_loading", path=args.ckpt)
    model, checkpoint_epoch, training_state = DiTTransformer2DWrapper.from_pretrained(
        save_directory=args.ckpt,
        device=device,
        return_training_state=True,
    )
    del training_state
    model.eval()
    logger.log_event(
        "checkpoint_loaded",
        path=args.ckpt,
        checkpoint_epoch=checkpoint_epoch,
    )

    logger.log_event("valid_dataset_initializing", train_data_dim_dir=args.train_data_dim_dir)
    dataset = PatchDataset(args.train_data_dim_dir)
    dim_channels = dataset[0].shape[0]
    validate_dim_latent_model_channels(model, latent_channels, dim_channels)
    rank_files = split_patch_files_for_rank(dataset)
    logger.log_event(
        "valid_dataset_initialized",
        dataset_size=len(dataset),
        dim_channels=int(dim_channels),
        total_files=len(dataset.patch_files),
        rank_files=len(rank_files),
        rank=distributed_mode.get_rank(),
        world_size=distributed_mode.get_world_size(),
        vae_latent_shape=list(vae_latent_shape),
    )

    solver = ODESolver(velocity_model=DimLatentVelocityModel(model).to(device))
    time_grid = torch.tensor([0.0, 1.0], device=device)
    file_summaries = []

    logger.log_event(
        "validation_started",
        checkpoint=args.ckpt,
        ckpt_vae=args.ckpt_vae,
        train_data_dim_dir=args.train_data_dim_dir,
        output_dir=str(output_dir),
        solver_step_size=args.solver_step_size,
        clip_recon="" if args.clip_recon is None else list(args.clip_recon),
        batch_size=args.batch_size,
        rank=distributed_mode.get_rank(),
        world_size=distributed_mode.get_world_size(),
    )

    with torch.inference_mode():
        for file_index, input_file in enumerate(rank_files):
            file_summary = reconstruct_dim_file(
                input_file=input_file,
                input_root=dataset.data_path,
                output_root=output_dir,
                vae=vae,
                solver=solver,
                time_grid=time_grid,
                device=device,
                args=args,
                logger=logger,
                latent_channels=latent_channels,
                latent_height=latent_height,
                latent_width=latent_width,
                file_index=file_index,
            )
            file_summaries.append(file_summary)

    local_summary = {
        "rank": distributed_mode.get_rank(),
        "num_files": len(file_summaries),
        "num_patches": int(sum(item["num_patches"] for item in file_summaries)),
        "files": file_summaries,
    }
    gathered_summaries = gather_validation_summaries(local_summary)
    flat_file_summaries = [
        file_summary
        for rank_summary in gathered_summaries
        for file_summary in rank_summary["files"]
    ]
    total_patches = int(sum(item["num_patches"] for item in flat_file_summaries))
    output_min = min(
        (item["output_min"] for item in flat_file_summaries if item["output_min"] is not None),
        default=None,
    )
    output_max = max(
        (item["output_max"] for item in flat_file_summaries if item["output_max"] is not None),
        default=None,
    )

    summary_path = output_dir / "summary.txt"
    if distributed_mode.get_rank() == 0:
        summary_lines = [
            f"checkpoint: {args.ckpt}",
            f"checkpoint_epoch: {checkpoint_epoch}",
            f"ckpt_vae: {args.ckpt_vae}",
            f"train_data_dim_dir: {args.train_data_dim_dir}",
            f"output_dir: {output_dir}",
            f"num_input_files: {len(dataset.patch_files)}",
            f"num_output_files: {len(flat_file_summaries)}",
            f"num_patches: {total_patches}",
            f"output_min: {'' if output_min is None else f'{output_min:.6g}'}",
            f"output_max: {'' if output_max is None else f'{output_max:.6g}'}",
            f"vae_latent_shape: {list(vae_latent_shape)}",
            f"solver_step_size: {args.solver_step_size}",
            f"clip_recon: {'' if args.clip_recon is None else list(args.clip_recon)}",
            "",
            "files:",
        ]
        for item in sorted(flat_file_summaries, key=lambda value: value["output_file"]):
            summary_lines.append(
                f"{item['output_file']} | patches={item['num_patches']} | "
                f"shape={item['output_shape']} | source={item['input_file']}"
            )
        summary_path.write_text("\n".join(summary_lines) + "\n")

    logger.log_event(
        "validation_finished",
        output_dir=str(output_dir),
        num_output_files=len(flat_file_summaries),
        num_patches=total_patches,
        output_min="" if output_min is None else output_min,
        output_max="" if output_max is None else output_max,
        summary=str(summary_path),
        rank=distributed_mode.get_rank(),
    )
    logger.close()

    if getattr(args, "distributed", False):
        if getattr(args, "dist_backend", None) == "nccl":
            distributed_mode.barrier([args.gpu])
        else:
            distributed_mode.barrier()
        distributed_mode.destroy()


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    if parsed_args.mode == "valid":
        run_valid(parsed_args)
    else:
        run_train(parsed_args)
