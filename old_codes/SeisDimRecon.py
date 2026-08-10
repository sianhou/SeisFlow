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
            "Train a conditional flow-matching model from noise to seismic patches with dimension coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  python SeisDimRecon.py "
            "--train_data_dir ./dataset256/train "
            "--train_data_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_dim_recon "
            "--model_arch DiT_T_4 --input_size 256 --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Valid:\n"
            "  python SeisDimRecon.py valid "
            "--ckpt ./output_dim_recon/run/checkpoint_epoch_01000 "
            "--train_data_dim_dir ./dataset256/valid_dim "
            "--output_dir ./output_dim_recon/valid "
            "--batch_size 32 --solver_step_size 0.05 --clip_recon -1 1 --device cuda\n"
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
        default=64,
        type=int,
        help="Height and width of the square training patches.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint directory. In train mode it resumes training; in valid mode it is required.",
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
    if not Path(args.train_data_dir).is_dir():
        raise FileNotFoundError(f"--train_data_dir must be a directory, got {args.train_data_dir}.")
    if not Path(args.train_data_dim_dir).is_dir():
        raise FileNotFoundError(f"--train_data_dim_dir must be a directory, got {args.train_data_dim_dir}.")
    if args.ckpt is not None and not Path(args.ckpt).is_dir():
        raise FileNotFoundError(f"--ckpt must be a checkpoint directory, got {args.ckpt}.")
    if args.input_size <= 0:
        raise ValueError("--input_size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("--num_epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.save_every_epochs <= 0:
        raise ValueError("--save_every_epochs must be positive.")
    if not 0.0 <= args.adam_beta1 < 1.0:
        raise ValueError("--adam_beta1 must be in [0, 1).")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("--adam_beta2 must be in [0, 1).")


def validate_valid_args(args):
    if args.ckpt is None:
        raise ValueError("--ckpt is required in valid mode.")
    if not Path(args.ckpt).is_dir():
        raise FileNotFoundError(f"--ckpt must be a checkpoint directory, got {args.ckpt}.")
    if not Path(args.train_data_dim_dir).is_dir():
        raise FileNotFoundError(f"--train_data_dim_dir must be a directory, got {args.train_data_dim_dir}.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.solver_step_size <= 0:
        raise ValueError("--solver_step_size must be positive.")
    if args.clip_recon is not None and args.clip_recon[0] >= args.clip_recon[1]:
        raise ValueError("--clip_recon MIN must be smaller than MAX.")


def validate_dim_model_channels(model, dim_channels):
    expected_model_in_channels = 1 + dim_channels
    model_in_channels = int(model.model.config.in_channels)
    model_out_channels = int(model.model.config.out_channels)
    if model_in_channels != expected_model_in_channels:
        raise ValueError(
            "Checkpoint input channel count does not match image + dim data: "
            f"model in_channels={model_in_channels}, dim_channels={dim_channels}, "
            f"expected={expected_model_in_channels}."
        )
    if model_out_channels != 1:
        raise ValueError(
            "Checkpoint output channel count does not match single-channel image patches: "
            f"model out_channels={model_out_channels}."
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


class DimVelocityModel(ModelWrapper):
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


def reconstruct_dim_file(
        input_file,
        input_root,
        output_root,
        solver,
        time_grid,
        device,
        args,
        logger=None,
        file_index=0,
        total_files=1,
):
    dim_array = np.load(input_file, mmap_mode="r")
    num_patches = int(dim_array.shape[0])
    output_file = get_reconstruction_output_file(input_file, input_root, output_root)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reconstructed_batches = []
    file_min = None
    file_max = None

    for batch_start in range(0, num_patches, args.batch_size):
        batch_end = min(batch_start + args.batch_size, num_patches)
        conditioning = make_conditioning_batch(dim_array, batch_start, batch_end, device)
        noise = torch.randn(
            (
                conditioning.shape[0],
                1,
                conditioning.shape[-2],
                conditioning.shape[-1],
            ),
            device=device,
            dtype=conditioning.dtype,
        )
        sampled = solver.sample(
            time_grid=time_grid,
            x_init=noise,
            return_intermediates=False,
            step_size=args.solver_step_size,
            cfg_scale=0.0,
            label=None,
            concat_conditioning={"concat_conditioning": conditioning},
        )
        reconstructed = sampled.detach().float().cpu()
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
        if logger is not None:
            logger.log_event(
                "batch_reconstructed",
                file=file_index + 1,
                total_files=total_files,
                file_name=Path(input_file).name,
                batch_start=batch_start,
                batch_end=batch_end,
                batch_size=batch_end - batch_start,
            )

    reconstructed_patches = torch.cat(reconstructed_batches, dim=0)
    reconstructed_array = restore_reconstruction_file_shape(reconstructed_patches, dim_array)
    np.save(output_file, reconstructed_array)

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "num_patches": num_patches,
        "output_shape": list(reconstructed_array.shape),
        "output_min": file_min,
        "output_max": file_max,
    }


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
        if logger is not None:
            logger.log_event(
                "batch_trained",
                epoch=epoch + 1,
                step=step + 1,
                total_steps=total_steps,
                batch_size=int(clean_images.shape[0]),
                loss=loss_value,
                running_loss=running_loss / max(running_steps, 1),
                lr=learning_rate,
                optimizer_step=int(should_step),
                grad_norm="" if grad_norm is None else float(grad_norm.detach().cpu()),
                clip_grad="" if clip_grad is None else clip_grad,
                step_time_sec=time.time() - step_start_time,
            )

    return epoch_loss / max(epoch_steps, 1)


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
    logger.log_event(
        "training_configured",
        dataset_size=len(dataset),
        num_batches=len(train_loader),
        dim_channels=int(dim_channels),
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

    logger = build_dist_logger(args)

    output_dir = logger.run_dir
    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    set_random_seed(seed)
    np.random.seed(seed)

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
    validate_dim_model_channels(model, dim_channels)
    rank_files = split_patch_files_for_rank(dataset)
    logger.log_event(
        "valid_dataset_initialized",
        dataset_size=len(dataset),
        dim_channels=int(dim_channels),
        total_files=len(dataset.patch_files),
        rank_files=len(rank_files),
        rank=distributed_mode.get_rank(),
        world_size=distributed_mode.get_world_size(),
    )

    solver = ODESolver(velocity_model=DimVelocityModel(model).to(device))
    time_grid = torch.tensor([0.0, 1.0], device=device)
    file_summaries = []

    logger.log_event(
        "validation_started",
        checkpoint=args.ckpt,
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
                solver=solver,
                time_grid=time_grid,
                device=device,
                args=args,
                logger=logger,
                file_index=file_index,
                total_files=len(rank_files),
            )
            file_summaries.append(file_summary)

    total_patches = int(sum(item["num_patches"] for item in file_summaries))
    output_min = min(
        (item["output_min"] for item in file_summaries if item["output_min"] is not None),
        default=None,
    )
    output_max = max(
        (item["output_max"] for item in file_summaries if item["output_max"] is not None),
        default=None,
    )

    logger.log_event(
        "validation_finished",
        output_dir=str(output_dir),
        num_output_files=len(file_summaries),
        num_patches=total_patches,
        output_min="" if output_min is None else output_min,
        output_max="" if output_max is None else output_max,
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
