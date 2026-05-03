import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from core.dataset import PatchDataset
from core.logging.logger import SimpleLogger2
from core.training import count_model_parameters, set_random_seed
from core.transforms.normalize import Normalize
from models.seismic_vae import SeismicSpatialVAE, kl_divergence


class NormalizePatch:
    def __init__(self):
        self.normalize = Normalize(mode="per_channel", method="abs")

    def __call__(self, patch):
        return self.normalize(patch.unsqueeze(0)).squeeze(0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Work Package 1 seismic spatial VAE on patch NPZ files."
    )
    parser.add_argument("--data_dir", default="./dataset_train_size256")
    parser.add_argument("--output_dir", default="./output_seismic_vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--adam_betas", nargs=2, type=float, default=[0.9, 0.95])
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument(
        "--clip_grad",
        default=1.0,
        type=float,
        help="Max gradient norm. Use 0 or a negative value to disable clipping.",
    )
    parser.add_argument("--checkpoint_interval", default=10, type=int)
    parser.add_argument("--log_id", default=None)
    parser.add_argument("--log_console", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--input_size", default=256, choices=[128, 256, 512], type=int)
    parser.add_argument("--input_channels", default=1, type=int)
    parser.add_argument(
        "--latent_channels",
        default=4,
        type=int,
        help="Custom latent channel count C for [B,C,latent_size,latent_size].",
    )
    parser.add_argument(
        "--latent_size",
        default=32,
        type=int,
        choices=[16, 32, 64],
        help="Custom spatial latent size N for a CxNxN latent map.",
    )
    parser.add_argument("--hidden_channels", default=32, type=int)
    parser.add_argument(
        "--channel_multipliers",
        nargs="*",
        type=int,
        default=None,
        help="Optional encoder channel multipliers. Defaults to powers of two.",
    )
    parser.add_argument("--autoencoder", action="store_true", help="Disable latent sampling; use z=mu.")
    parser.add_argument(
        "--kl_weight",
        default=0.0,
        type=float,
        help="KL loss weight. Use 0 to disable KL regularization.",
    )
    return parser


def build_dataloader(args):
    transform = NormalizePatch() if args.normalize else None
    dataset = PatchDataset(args.data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    return dataset, dataloader


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


def batch_metrics(recon, target):
    with torch.no_grad():
        mae = F.l1_loss(recon, target)
        mse = F.mse_loss(recon, target)
        rmse = torch.sqrt(mse.clamp_min(1e-12))
        psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(max(float(mse.item()), 1e-12))
        return {
            "mae": float(mae.item()),
            "mse": float(mse.item()),
            "rmse": float(rmse.item()),
            "psnr": psnr,
        }


def cuda_max_allocated_mb(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def training_mode_name(args):
    if args.autoencoder and args.kl_weight > 0:
        return "autoencoder_kl_regularized"
    if args.autoencoder:
        return "autoencoder"
    if args.kl_weight > 0:
        return "weak_kl_vae"
    return "sampled_vae_no_kl"


def save_checkpoint(model, optimizer, args, epoch, output_dir):
    checkpoint_path = Path(output_dir) / f"seismic_vae_epoch_{epoch + 1:05d}.pth"
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config.__dict__,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def validate_args(args):
    if args.input_channels != 1:
        raise ValueError("WP1 requires --input_channels 1.")
    if args.latent_channels <= 0:
        raise ValueError("latent_channels must be positive.")
    if args.latent_size not in {16, 32, 64}:
        raise ValueError("latent_size must be one of: 16, 32, 64.")
    if args.input_size % args.latent_size != 0:
        raise ValueError("input_size must be divisible by latent_size.")
    if args.kl_weight < 0:
        raise ValueError("kl_weight must be non-negative.")
    compression = args.input_size // args.latent_size
    if compression < 1 or compression & (compression - 1) != 0:
        raise ValueError("input_size / latent_size must be a power of two.")


def main(args):
    validate_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger2(
        output_dir=args.output_dir,
        log_id=args.log_id,
        overwrite=True,
        console=args.log_console,
        logs=[
            "epoch",
            "step",
            "global_step",
            "optimizer_step",
            "batch_size",
            "loss",
            "mse_loss",
            "kl_loss",
            "kl_weight",
            "mae",
            "mse",
            "rmse",
            "psnr",
            "lr",
            "grad_norm",
            "nonfinite_grad_norm",
            "cuda_max_allocated_mb",
            "step_time_sec",
            "total_time_sec",
            "samples_per_sec",
        ],
    )
    logger.log_event("script_started", job_dir=os.path.dirname(os.path.realpath(__file__)))
    logger.log_argparse_params(args)

    device = torch.device(args.device)
    set_random_seed(args.seed)

    dataset, dataloader = build_dataloader(args)

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

    total_params, trainable_params, frozen_params = count_model_parameters(model)
    logger.log_system_info(
        package_names=[
            "torch",
            "torchvision",
            "torchmetrics",
            "numpy",
            "matplotlib",
            "segyio",
        ]
    )
    logger.log_global_params(
        {
            "task": "work_package_1_seismic_spatial_vae",
            "data_dir": args.data_dir,
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "normalization": "per_channel_abs" if args.normalize else "as_stored",
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
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "adam_betas": args.adam_betas,
            "grad_accum_steps": args.grad_accum_steps,
            "clip_grad": args.clip_grad,
            "loss": "mse_plus_kl",
            "kl_weight": args.kl_weight,
            "amp": False,
            "model": str(model),
        }
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.adam_betas),
    )
    run_output_dir = logger.run_dir
    config_path = run_output_dir / "seismic_vae_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model_config": model.config.__dict__}, f, indent=2)
    logger.log_event("config_saved", path=str(config_path))

    logger.log_event("training_started")
    start_time = time.time()
    global_step = 0
    optimizer_step = 0
    for epoch in range(args.num_epochs):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            step_start_time = time.time()
            global_step += 1
            if batch.shape[1] != args.input_channels:
                raise ValueError(
                    f"Expected batch channel count {args.input_channels}, got {batch.shape[1]}."
                )
            if batch.shape[-2:] != (args.input_size, args.input_size):
                raise ValueError(
                    f"Expected patches with shape {args.input_size}x{args.input_size}, "
                    f"got {tuple(batch.shape[-2:])}. Rebuild patches or set --input_size."
                )

            clean_images = batch.to(device, non_blocking=True)
            outputs = model(clean_images)
            losses = compute_loss(outputs, clean_images, args)
            metrics = batch_metrics(outputs["recon"], clean_images)

            loss = losses["total"] / args.grad_accum_steps
            should_step = (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(dataloader)
            loss.backward()

            grad_norm = None
            nonfinite_grad_norm = 0
            if should_step:
                if args.clip_grad > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.clip_grad,
                    )
                else:
                    grad_norms = [
                        torch.linalg.vector_norm(parameter.grad.detach(), 2)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                    if grad_norms:
                        grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms), 2)
                    else:
                        grad_norm = torch.zeros((), device=device)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

            grad_norm_value = ""
            if grad_norm is not None:
                grad_norm_value = float(grad_norm.detach().cpu())
                nonfinite_grad_norm = 0 if math.isfinite(grad_norm_value) else 1

            step_time = time.time() - step_start_time
            logger.log_train(
                epoch=epoch + 1,
                step=step + 1,
                global_step=global_step,
                optimizer_step=optimizer_step,
                batch_size=int(batch.shape[0]),
                loss=float(losses["total"].detach().cpu()),
                mse_loss=float(losses["mse"].detach().cpu()),
                kl_loss=float(losses["kl"].detach().cpu()),
                kl_weight=args.kl_weight,
                mae=metrics["mae"],
                mse=metrics["mse"],
                rmse=metrics["rmse"],
                psnr=metrics["psnr"],
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=grad_norm_value,
                nonfinite_grad_norm=nonfinite_grad_norm,
                cuda_max_allocated_mb=cuda_max_allocated_mb(device),
                step_time_sec=step_time,
                total_time_sec=time.time() - start_time,
                samples_per_sec=float(batch.shape[0]) / max(step_time, 1e-12),
            )

        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.num_epochs:
            checkpoint_path = save_checkpoint(model, optimizer, args, epoch, run_output_dir)
            logger.log_event(
                "checkpoint_saved",
                epoch=epoch + 1,
                path=str(checkpoint_path),
            )

    total_time = time.time() - start_time
    logger.log_event("training_finished", total_time_sec=total_time, run_dir=str(run_output_dir))
    logger.close()


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
