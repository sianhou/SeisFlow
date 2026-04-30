import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".matplotlib_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from core.dataset import SegyDataset
from models.seismic_vae import SeismicSpatialVAE


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate a Seismic VAE checkpoint on random SEG-Y shot crops."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a seismic_vae_epoch_*.pth checkpoint.",
    )
    parser.add_argument(
        "--segy",
        required=True,
        help="SEG-Y file used for random shot crop validation.",
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: <checkpoint_parent>/valid_<checkpoint_stem>.",
    )
    parser.add_argument(
        "--slice",
        nargs=2,
        type=int,
        default=[0, 1501],
        help="Optional sample-axis slice before random crop. Use 0 0 to disable.",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Reserved for future dataloader validation. Current script samples directly.",
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config")
    if config is None:
        config = checkpoint.get("args", {})
        config = {
            "input_channels": config.get("input_channels", 1),
            "output_channels": 1,
            "latent_channels": config.get("latent_channels", 4),
            "input_size": config.get("input_size", 256),
            "latent_size": config.get("latent_size", 32),
            "hidden_channels": config.get("hidden_channels", 32),
            "channel_multipliers": config.get("channel_multipliers"),
            "use_vae": not config.get("autoencoder", False),
        }
    model = SeismicSpatialVAE.from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, config


def maybe_slice_shot(shot, slice_range):
    start, end = slice_range
    if 0 <= start < end:
        return shot[:, start:end]
    return shot


def ensure_min_size(shot, crop_size):
    height, width = shot.shape
    if height >= crop_size and width >= crop_size:
        return shot

    tensor = torch.from_numpy(shot).float().unsqueeze(0).unsqueeze(0)
    target_h = max(height, crop_size)
    target_w = max(width, crop_size)
    resized = F.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).numpy()


def random_crop(shot, crop_size):
    shot = ensure_min_size(shot, crop_size)
    height, width = shot.shape
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    crop = shot[top: top + crop_size, left: left + crop_size]
    return crop.astype(np.float32), top, left, shot.shape


def compute_metrics(recon, target):
    diff = recon - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = math.sqrt(max(mse, 1e-12))
    psnr_range = max(float(np.max(target) - np.min(target)), 1e-12)
    psnr = 20.0 * math.log10(psnr_range) - 10.0 * math.log10(max(mse, 1e-12))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
    }


def normalize_batch_abs(raw_batch):
    scales = np.max(np.abs(raw_batch), axis=(1, 2)).astype(np.float32)
    scales = np.maximum(scales, 1e-12)
    normalized = raw_batch / scales[:, None, None]
    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)
    return normalized, scales


def plot_sample(raw, recon, diff, metrics, output_path, title):
    vlim = max(float(np.max(np.abs(raw))), float(np.max(np.abs(recon))), 1e-6)
    diff_vlim = max(float(np.max(np.abs(diff))), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    images = [
        (raw, "input", "seismic", -vlim, vlim),
        (recon, "recon", "seismic", -vlim, vlim),
        (diff, "diff", "seismic", -diff_vlim, diff_vlim),
    ]
    for axis, (image, name, cmap, vmin, vmax) in zip(axes, images):
        im = axis.imshow(image.T, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")
        fig.colorbar(im, ax=axis, shrink=0.8)

    fig.suptitle(
        f"{title} | mse={metrics['mse']:.6g}, mae={metrics['mae']:.6g}, "
        f"psnr={metrics['psnr']:.3f}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_batch_grid(raw_batch, recon_batch, diff_batch, output_path):
    batch_size = raw_batch.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(10, max(3, batch_size * 2.4)))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    vlim = max(float(np.max(np.abs(raw_batch))), float(np.max(np.abs(recon_batch))), 1e-6)
    diff_vlim = max(float(np.max(np.abs(diff_batch))), 1e-6)

    for idx in range(batch_size):
        row_items = [
            (raw_batch[idx], "input", -vlim, vlim),
            (recon_batch[idx], "recon", -vlim, vlim),
            (diff_batch[idx], "diff", -diff_vlim, diff_vlim),
        ]
        for axis, (image, title, vmin, vmax) in zip(axes[idx], row_items):
            axis.imshow(image.T, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax)
            axis.set_title(f"{idx} {title}", fontsize=9)
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint_path = Path(args.checkpoint)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent / f"valid_{checkpoint_path.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, config = load_model(checkpoint_path, device)
    crop_size = int(config["input_size"])
    dataset = SegyDataset(args.segy)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if len(dataset) == 0:
        raise ValueError(f"No shots found in {args.segy}.")

    sample_count = min(args.batch_size, len(dataset))
    shot_indices = random.sample(range(len(dataset)), sample_count)

    raw_patches = []
    metadata = []
    for sample_idx, shot_index in enumerate(shot_indices):
        shot_tensor = dataset[shot_index]
        shot = shot_tensor[0].detach().cpu().numpy().astype(np.float32)
        shot = maybe_slice_shot(shot, args.slice)
        crop, top, left, effective_shape = random_crop(shot, crop_size)
        raw_patches.append(crop)
        metadata.append(
            {
                "sample_idx": sample_idx,
                "shot_index": shot_index,
                "shot_id": dataset.shot_keys[shot_index],
                "crop_top": top,
                "crop_left": left,
                "effective_height": effective_shape[0],
                "effective_width": effective_shape[1],
            }
        )

    raw_batch = np.stack(raw_patches, axis=0)
    input_batch, norm_scales = normalize_batch_abs(raw_batch)
    input_tensor = torch.from_numpy(input_batch).float().unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        recon_normalized_batch = outputs["recon"].squeeze(1).detach().cpu().numpy()

    recon_batch = recon_normalized_batch * norm_scales[:, None, None]
    diff_batch = recon_batch - raw_batch
    metrics_rows = []
    for idx in range(sample_count):
        metrics = compute_metrics(recon_batch[idx], raw_batch[idx])
        row = {**metadata[idx], "norm_scale": float(norm_scales[idx]), **metrics}
        metrics_rows.append(row)
        plot_sample(
            raw_batch[idx],
            recon_batch[idx],
            diff_batch[idx],
            metrics,
            output_dir / f"sample_{idx:03d}_shot_{metadata[idx]['shot_index']:04d}.png",
            title=f"shot={metadata[idx]['shot_index']} crop=({metadata[idx]['crop_top']},{metadata[idx]['crop_left']})",
        )

    plot_batch_grid(raw_batch, recon_batch, diff_batch, output_dir / "batch_reconstruction.png")

    np.savez_compressed(
        output_dir / "valid_samples.npz",
        raw=raw_batch,
        input_normalized=input_batch,
        recon_normalized=recon_normalized_batch,
        recon=recon_batch,
        diff=diff_batch,
        norm_scales=norm_scales,
        shot_indices=np.array(shot_indices, dtype=np.int64),
    )

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "sample_idx",
        "shot_index",
        "shot_id",
        "crop_top",
        "crop_left",
        "effective_height",
        "effective_width",
        "norm_scale",
        "mse",
        "mae",
        "rmse",
        "psnr",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "segy": args.segy,
        "device": str(device),
        "batch_size": sample_count,
        "input_size": crop_size,
        "input_normalization": "per_patch_abs",
        "inverse_normalize_reconstruction": True,
        "metric_domain": "raw_amplitude",
        "psnr_range": "target_max_minus_min",
        "slice": args.slice,
        "mean_mse": float(np.mean([row["mse"] for row in metrics_rows])),
        "mean_mae": float(np.mean([row["mae"] for row in metrics_rows])),
        "mean_rmse": float(np.mean([row["rmse"] for row in metrics_rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in metrics_rows])),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved: {output_dir}")
    print(
        f"mean_mse={summary['mean_mse']:.6g} "
        f"mean_mae={summary['mean_mae']:.6g} "
        f"mean_psnr={summary['mean_psnr']:.3f}"
    )
    return output_dir


def main():
    args = build_parser().parse_args()
    validate(args)


if __name__ == "__main__":
    main()
