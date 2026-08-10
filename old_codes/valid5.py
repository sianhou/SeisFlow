import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from core.dataset import SegyDataset
from core.logging.logger import SimpleLogger2
from core.masks.row_mask import generate_random_row_mask
from core.metrics import compute_psnr
from core.patching import TensorPatchProcessor
from core.training import set_random_seed
from core.transforms import AbsNormalize
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.wrapper import DIT_TRANSFORMER_2D_CONFIGS, DiTTransformer2DWrapper


class ConditionalVelocityModel(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cfg_scale: float,
            label: torch.Tensor,
            concat_conditioning,
    ) -> torch.Tensor:
        del cfg_scale, label

        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])

        with torch.inference_mode():
            with torch.amp.autocast(
                    device_type=x.device.type,
                    enabled=x.device.type == "cuda",
            ):
                result = self.model(x, t, extra=concat_conditioning)

        return result.to(dtype=torch.float32)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate all seismic shots with a train5.py checkpoint directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--segy", required=True, help="SEG-Y file used for validation.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory produced by train5.py, e.g. checkpoint_epoch_00050.",
    )
    parser.add_argument(
        "--output_dir",
        default="./valid5_output",
        help="Directory used to save figures and NPY files.",
    )
    parser.add_argument("--log_id", default=None, help="Optional validation run directory name under output_dir.")
    parser.add_argument("--log_console", action="store_true", help="Also print SimpleLogger2 output to stdout.")
    parser.add_argument("--device", default="cuda", help="Sampling device.")
    parser.add_argument(
        "--model_arch",
        choices=sorted(DIT_TRANSFORMER_2D_CONFIGS.keys()),
        default=None,
        help="Model architecture used during training. Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Fixed random row-mask ratio.")
    parser.add_argument("--patch_size", default=None, type=int, help="Patch size used to train train5.py.")
    parser.add_argument("--overlap_size", default=32, type=int, help="Overlap size used when extracting patches.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Number of patches sampled at once during validation.",
    )
    parser.add_argument(
        "--shot_interval",
        default=10,
        type=int,
        help="Validate every N shots, starting from shot 0.",
    )
    parser.add_argument(
        "--clip_vmin",
        default=None,
        type=float,
        help="Optional lower clipping bound for saved PNG figures only.",
    )
    parser.add_argument(
        "--clip_vmax",
        default=None,
        type=float,
        help="Optional upper clipping bound for saved PNG figures only.",
    )
    parser.add_argument(
        "--slice",
        nargs=2,
        type=int,
        default=[0, 1501],
        metavar=("START", "END"),
        help="Optional sample-axis slice for saved PNG figures only. Use 0 0 to disable.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
        help="Optional resize for saved PNG figures only. Use 0 0 to disable.",
    )
    parser.add_argument("--solver_step_size", default=0.05, type=float, help="Euler solver step size.")
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed used for mask generation and sampling noise.",
    )
    return parser


def load_train5_checkpoint(checkpoint_dir, device):
    wrapper, epoch, training_state = DiTTransformer2DWrapper.from_pretrained(
        save_directory=checkpoint_dir,
        device=device,
        return_training_state=True,
    )
    return wrapper, epoch, training_state


def resolve_checkpoint_args(args, training_state):
    checkpoint_args = training_state.get("args") or {}

    checkpoint_arch = checkpoint_args.get("model_arch")
    if args.model_arch is None:
        if checkpoint_arch is None:
            raise ValueError(
                "--model_arch is required because the checkpoint does not contain model_arch."
            )
        args.model_arch = checkpoint_arch
    if checkpoint_arch is not None and args.model_arch != checkpoint_arch:
        raise ValueError(
            f"--model_arch={args.model_arch} does not match checkpoint model_arch={checkpoint_arch}."
        )

    checkpoint_input_size = checkpoint_args.get("input_size")
    if args.patch_size is None:
        if checkpoint_input_size is None:
            raise ValueError(
                "--patch_size is required because the checkpoint does not contain input_size."
            )
        args.patch_size = int(checkpoint_input_size)
    if checkpoint_input_size is not None and args.patch_size != int(checkpoint_input_size):
        raise ValueError(
            f"--patch_size={args.patch_size} does not match checkpoint input_size={checkpoint_input_size}."
        )

    return checkpoint_args


def validate_args(args):
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"--checkpoint must be a train5.py checkpoint directory, got {checkpoint_dir}."
        )
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.overlap_size < 0:
        raise ValueError("--overlap_size must be non-negative.")
    if args.overlap_size >= args.patch_size:
        raise ValueError("--overlap_size must be smaller than --patch_size.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.shot_interval <= 0:
        raise ValueError("--shot_interval must be positive.")
    if not (0.0 <= args.mask_ratio < 1.0):
        raise ValueError("--mask_ratio must satisfy 0 <= value < 1.")
    if (
            args.clip_vmin is not None
            and args.clip_vmax is not None
            and args.clip_vmin > args.clip_vmax
    ):
        raise ValueError("--clip_vmin must be less than or equal to --clip_vmax.")
    if args.slice[0] < 0 or args.slice[1] < 0:
        raise ValueError("--slice values must be non-negative. Use 0 0 to disable.")
    if args.slice != [0, 0] and args.slice[0] >= args.slice[1]:
        raise ValueError("--slice START must be smaller than END, or use 0 0 to disable.")
    if args.resize[0] < 0 or args.resize[1] < 0:
        raise ValueError("--resize values must be non-negative. Use 0 0 to disable.")
    if args.resize != [0, 0] and (args.resize[0] == 0 or args.resize[1] == 0):
        raise ValueError("--resize must be HEIGHT WIDTH with both > 0, or 0 0 to disable.")


def compute_patch_batch_metrics(clean_batch: torch.Tensor, recon_batch: torch.Tensor):
    clean_np = clean_batch.detach().cpu().numpy()
    recon_np = recon_batch.detach().cpu().numpy()

    psnr_values = []
    mae_values = []
    for clean_patch, recon_patch in zip(clean_np, recon_np):
        dynamic_range = max(float(clean_patch.max() - clean_patch.min()), 1e-6)
        psnr_values.append(compute_psnr(clean_patch, recon_patch, max_pixel=dynamic_range))
        mae_values.append(float(np.mean(np.abs(clean_patch - recon_patch))))

    return float(np.mean(psnr_values)), float(np.mean(mae_values))


def sample_one_shot(
        sample: torch.Tensor,
        solver: ODESolver,
        patch_processor: TensorPatchProcessor,
        time_grid: torch.Tensor,
        args,
        logger,
        shot_index: int,
):
    mask_ratio = float(args.mask_ratio)
    mask = generate_random_row_mask(sample, missing_ratio=mask_ratio)
    missed = mask * sample

    clean_patches, positions, original_shape = patch_processor.extract_overlapping_patches_2d(
        sample,
        patch_size=(args.patch_size, args.patch_size),
        overlap=(args.overlap_size, args.overlap_size),
    )
    mask_patches, _, _ = patch_processor.extract_overlapping_patches_2d(
        mask,
        patch_size=(args.patch_size, args.patch_size),
        overlap=(args.overlap_size, args.overlap_size),
    )

    clean_patches = clean_patches.squeeze(0)
    mask_patches = mask_patches.squeeze(0)

    normalizer = AbsNormalize(per_channel=True)
    normalized_clean_patches, patch_scales = normalizer.run(clean_patches)
    normalized_missed_patches = normalized_clean_patches * mask_patches

    num_patches = int(normalized_clean_patches.shape[0])
    num_patch_batches = (num_patches + args.batch_size - 1) // args.batch_size

    reconstructed_normalized_batches = []
    for batch_index, start in enumerate(range(0, num_patches, args.batch_size), start=0):
        end = min(start + args.batch_size, num_patches)
        missed_batch = normalized_missed_patches[start:end]
        mask_batch = mask_patches[start:end]
        step_start_time = time.time()
        sampled_batch = solver.sample(
            time_grid=time_grid,
            x_init=torch.randn_like(missed_batch),
            return_intermediates=False,
            step_size=args.solver_step_size,
            cfg_scale=0.0,
            label=None,
            concat_conditioning={
                "concat_conditioning": torch.cat((missed_batch, mask_batch), dim=1)
            },
        )
        reconstructed_normalized_batch = missed_batch + (1.0 - mask_batch) * sampled_batch
        reconstructed_normalized_batches.append(reconstructed_normalized_batch)
        reconstructed_batch = reconstructed_normalized_batch * patch_scales[start:end]
        batch_psnr, batch_mae = compute_patch_batch_metrics(
            clean_patches[start:end],
            reconstructed_batch,
        )
        logger.log_valid(
            shot=shot_index,
            step=batch_index,
            num_batch=num_patch_batches,
            batch_start=start,
            batch_end=end,
            mask_ratio=mask_ratio,
            psnr=batch_psnr,
            mae=batch_mae,
            step_time_sec=time.time() - step_start_time,
        )

    reconstructed_normalized_patches = torch.cat(reconstructed_normalized_batches, dim=0)

    # Use clean patch statistics to map predictions back for evaluation/visualization.
    reconstructed_patches = (reconstructed_normalized_patches * patch_scales).unsqueeze(0)

    reconstructed_sample = patch_processor.reconstruct_from_overlapping_patches_2d(
        reconstructed_patches,
        positions,
        original_shape,
    )
    return {
        "mask_ratio": mask_ratio,
        "raw": sample,
        "mask": mask,
        "missed": missed,
        "recon": reconstructed_sample,
        "num_patches": num_patches,
        "num_patch_batches": num_patch_batches,
    }


def prepare_plot_image(image: np.ndarray, args, clip=False):
    plot_image = image
    if clip and (args.clip_vmin is not None or args.clip_vmax is not None):
        vmin = args.clip_vmin if args.clip_vmin is not None else float(np.min(plot_image))
        vmax = args.clip_vmax if args.clip_vmax is not None else float(np.max(plot_image))
        plot_image = np.clip(plot_image, vmin, vmax)
    if args.slice != [0, 0]:
        plot_image = plot_image[..., args.slice[0]:args.slice[1]]
    if args.resize != [0, 0]:
        tensor = torch.from_numpy(np.ascontiguousarray(plot_image)).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(args.resize[0], args.resize[1]),
            mode="bilinear",
            align_corners=False,
        )
        plot_image = tensor.squeeze(0).squeeze(0).numpy()
    return plot_image


def plot_shot_comparison(raw, missed, recon, diff, output_path: Path, shot_index: int):
    seismic_vlim = max(
        float(np.max(np.abs(raw))),
        float(np.max(np.abs(missed))),
        float(np.max(np.abs(recon))),
        1e-6,
    )
    diff_vlim = max(float(np.max(np.abs(diff))), 1e-6)
    panels = [
        (raw, "raw", -seismic_vlim, seismic_vlim),
        (missed, "missed", -seismic_vlim, seismic_vlim),
        (recon, "recon", -seismic_vlim, seismic_vlim),
        (diff, "diff", -diff_vlim, diff_vlim),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, (image, title, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image.T, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"shot_index {shot_index}", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_shot_outputs(result, shot_index: int, shot_dir: Path, args):
    shot_dir.mkdir(parents=True, exist_ok=True)

    raw = result["raw"][0, 0].detach().cpu().numpy()
    mask = result["mask"][0, 0].detach().cpu().numpy()
    missed = result["missed"][0, 0].detach().cpu().numpy()
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    recon = result["recon"][0, 0].detach().cpu().clamp(raw_min, raw_max).numpy()
    diff = raw - recon

    np.save(shot_dir / "raw.npy", raw)
    np.save(shot_dir / "mask.npy", mask)
    np.save(shot_dir / "missed.npy", missed)
    np.save(shot_dir / "recon.npy", recon)
    np.save(shot_dir / "diff.npy", diff)
    np.save(
        shot_dir / "mask_ratio.npy",
        np.array(result["mask_ratio"], dtype=np.float32),
    )

    raw_plot = prepare_plot_image(raw, args, clip=True)
    missed_plot = prepare_plot_image(missed, args, clip=True)
    recon_plot = prepare_plot_image(recon, args, clip=True)
    diff_plot = raw_plot - recon_plot

    plot_path = shot_dir.parent / f"{shot_index}.png"
    plot_shot_comparison(raw_plot, missed_plot, recon_plot, diff_plot, plot_path, shot_index)

    dynamic_range = max(float(raw.max() - raw.min()), 1e-6)
    psnr = compute_psnr(raw, recon, max_pixel=dynamic_range)
    mae = float(np.mean(np.abs(diff)))
    return shot_dir, plot_path, psnr, mae


def main(args):
    device = torch.device(args.device)
    model, checkpoint_epoch, training_state = load_train5_checkpoint(args.checkpoint, device)
    checkpoint_args = resolve_checkpoint_args(args, training_state)
    validate_args(args)

    logger = SimpleLogger2(
        output_dir=args.output_dir,
        log_id=args.log_id,
        overwrite=True,
        console=args.log_console,
        logs=[
            "shot",
            "step",
            "num_batch",
            "batch_start",
            "batch_end",
            "mask_ratio",
            "psnr",
            "mae",
            "step_time_sec",
        ],
    )
    output_dir = logger.run_dir
    logger.log_event(
        "script_started",
        job_dir=os.path.dirname(os.path.realpath(__file__)),
        log_file=logger.log_file,
    )
    logger.log_info_block("ARGPARSE PARAMETERS", args)
    logger.log_info_block(
        "CHECKPOINT PARAMETERS",
        {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_model_arch": checkpoint_args.get("model_arch", ""),
            "checkpoint_input_size": checkpoint_args.get("input_size", ""),
        },
    )

    set_random_seed(args.seed)

    logger.log_event("model_initialized", model_arch=args.model_arch)
    model.eval()

    wrapped_model = ConditionalVelocityModel(model).to(device)
    solver = ODESolver(velocity_model=wrapped_model)
    patch_processor = TensorPatchProcessor()
    time_grid = torch.tensor([0.0, 1.0], device=device)

    dataset = SegyDataset(args.segy)
    shot_indices = list(range(0, len(dataset), args.shot_interval))
    logger.log_event(
        "dataset_initialized",
        segy=args.segy,
        num_shots=len(dataset),
        shot_interval=args.shot_interval,
        selected_shots=len(shot_indices),
    )

    logger.log_event("sampling_started", shot_interval=args.shot_interval, segy=args.segy)
    shot_metrics = []
    for shot_index in shot_indices:
        logger.log_event("shot_started", shot_index=shot_index)
        sample = dataset[shot_index][0].unsqueeze(0).unsqueeze(0).to(device)
        result = sample_one_shot(
            sample=sample,
            solver=solver,
            patch_processor=patch_processor,
            time_grid=time_grid,
            args=args,
            logger=logger,
            shot_index=shot_index,
        )
        shot_dir, plot_path, psnr, mae = save_shot_outputs(
            result,
            shot_index,
            output_dir / f"shot_{shot_index:04d}",
            args,
        )
        shot_metrics.append((psnr, mae))
        logger.log_valid(
            shot=shot_index,
            step=result["num_patch_batches"],
            num_batch=result["num_patch_batches"],
            batch_start="-",
            batch_end="-",
            mask_ratio=result["mask_ratio"],
            psnr=psnr,
            mae=mae,
            step_time_sec="-",
        )
        logger.log_event(
            "shot_finished",
            shot_index=shot_index,
            mask_ratio=result["mask_ratio"],
            psnr=psnr,
            mae=mae,
            num_patches=result["num_patches"],
            num_patch_batches=result["num_patch_batches"],
            saved=str(shot_dir),
            plot=str(plot_path),
        )
        del sample, result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_psnr = float(np.mean([item[0] for item in shot_metrics])) if shot_metrics else float("nan")
    mean_mae = float(np.mean([item[1] for item in shot_metrics])) if shot_metrics else float("nan")
    logger.log_event(
        "validation_finished",
        shot_interval=args.shot_interval,
        selected_shots=len(shot_indices),
        mean_psnr=mean_psnr,
        mean_mae=mean_mae,
        output_dir=str(output_dir),
    )
    logger.close()


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
