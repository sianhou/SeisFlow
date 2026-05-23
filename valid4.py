import argparse
import os
from pathlib import Path

import numpy as np
import torch

from core.dataset import SegyDataset
from core.logging.logger import SimpleLogger2
from core.masks.row_mask import generate_random_row_mask
from core.metrics import compute_psnr
from core.patching import TensorPatchProcessor
from core.training import set_random_seed
from core.transforms import AbsNormalize, ClipFirstChannel
from core.visualization import plot_seismic_grid
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.reconstruction_model_configs import MODEL_CONFIGS, build_model


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
        description="Validate all seismic shots with a train4.py checkpoint.",
    )
    parser.add_argument("--segy", required=True, help="SEG-Y file used for validation.", )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train4.py.", )
    parser.add_argument("--output_dir", default="./valid4_output",
                        help="Directory used to save figures and NPY files.", )
    parser.add_argument("--log_id", default=None, help="Optional validation run directory name under output_dir.")
    parser.add_argument("--log_console", action="store_true", help="Also print SimpleLogger2 output to stdout.")
    parser.add_argument("--device", default="cuda", help="Sampling device.", )
    parser.add_argument(
        "--model_arch",
        choices=sorted(MODEL_CONFIGS.keys()),
        default=None,
        help="Model architecture used during training. Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Fixed random row-mask ratio.", )
    parser.add_argument("--patch_size", default=64, type=int, help="Patch size used to train the patch dataset.", )
    parser.add_argument("--overlap_size", default=32, type=int, help="Overlap size used when extracting patches.", )
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Number of patches sampled at once during validation.")
    parser.add_argument("--shot_index", default=0, type=int, help="Shot index to validate.", )
    parser.add_argument("--clip_vmin", default=None, type=float,
                        help="Optional lower clipping bound for the seismic trace channel.")
    parser.add_argument("--clip_vmax", default=None, type=float,
                        help="Optional upper clipping bound for the seismic trace channel.")
    parser.add_argument("--solver_step_size", default=0.05, type=float, help="Euler solver step size.", )
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed used for mask generation and sampling noise.", )
    return parser


def load_train4_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint
    return {
        "model": checkpoint,
        "model_arch": None,
        "model_config": None,
        "args": {},
    }


def resolve_model_arch(args, checkpoint):
    checkpoint_arch = checkpoint.get("model_arch")
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
    return args.model_arch


def validate_args(args):
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.overlap_size < 0:
        raise ValueError("--overlap_size must be non-negative.")
    if args.overlap_size >= args.patch_size:
        raise ValueError("--overlap_size must be smaller than --patch_size.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if not (0.0 <= args.mask_ratio < 1.0):
        raise ValueError("--mask_ratio must satisfy 0 <= value < 1.")
    if (
            args.clip_vmin is not None
            and args.clip_vmax is not None
            and args.clip_vmin > args.clip_vmax
    ):
        raise ValueError("--clip_vmin must be less than or equal to --clip_vmax.")

    model_config = MODEL_CONFIGS[args.model_arch]
    expected_input_size = model_config.get("input_size")
    if expected_input_size is not None and args.patch_size != expected_input_size:
        raise ValueError(
            f"{args.model_arch} expects patch_size={expected_input_size}, "
            f"but got {args.patch_size}."
        )


def sample_one_shot(
        sample: torch.Tensor,
        solver: ODESolver,
        patch_processor: TensorPatchProcessor,
        time_grid: torch.Tensor,
        args,
        logger,
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
    logger.log_event(
        "patch_sampling_started",
        num_patches=num_patches,
        batch_size=args.batch_size,
        num_patch_batches=num_patch_batches,
    )
    reconstructed_normalized_batches = []
    for batch_index, start in enumerate(
            range(0, num_patches, args.batch_size),
            start=1,
    ):
        end = min(start + args.batch_size, num_patches)
        logger.log_event(
            "patch_batch_started",
            batch_index=batch_index,
            num_patch_batches=num_patch_batches,
            patch_start=start,
            patch_end=end,
            batch_size=end - start,
        )
        missed_batch = normalized_missed_patches[start:end]
        mask_batch = mask_patches[start:end]
        sampled_batch = solver.sample(
            time_grid=time_grid,
            x_init=torch.randn_like(missed_batch),
            return_intermediates=False,
            step_size=args.solver_step_size,
            cfg_scale=0.0,
            label=None,
            concat_conditioning={
                "concat_conditioning": torch.cat(
                    (missed_batch, mask_batch),
                    dim=1,
                )
            },
        )
        reconstructed_normalized_batches.append(
            missed_batch + (1.0 - mask_batch) * sampled_batch
        )
        logger.log_event(
            "patch_batch_finished",
            batch_index=batch_index,
            num_patch_batches=num_patch_batches,
            patch_start=start,
            patch_end=end,
        )

    reconstructed_normalized_patches = torch.cat(
        reconstructed_normalized_batches,
        dim=0,
    )
    logger.log_event("patch_sampling_finished", num_patches=num_patches)

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
    }


def save_shot_outputs(result, shot_index: int, shot_dir: Path):
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

    plot_seismic_grid(raw[None, ...], shot_dir / "raw.png", title=f"shot {shot_index} raw", size=1)
    plot_seismic_grid(mask[None, ...], shot_dir / "mask.png", title=f"shot {shot_index} mask", size=1)
    plot_seismic_grid(missed[None, ...], shot_dir / "missed.png", title=f"shot {shot_index} missed", size=1)
    plot_seismic_grid(recon[None, ...], shot_dir / "recon.png", title=f"shot {shot_index} recon", size=1)
    plot_seismic_grid(diff[None, ...], shot_dir / "diff.png", title=f"shot {shot_index} diff", size=1)

    dynamic_range = max(float(raw.max() - raw.min()), 1e-6)
    psnr = compute_psnr(raw, recon, max_pixel=dynamic_range)
    mae = float(np.mean(np.abs(diff)))
    return shot_dir, psnr, mae


def main(args):
    device = torch.device(args.device)
    checkpoint = load_train4_checkpoint(args.checkpoint, device)
    resolve_model_arch(args, checkpoint)
    validate_args(args)

    logger = SimpleLogger2(
        output_dir=args.output_dir,
        log_id=args.log_id,
        overwrite=True,
        console=args.log_console,
    )
    output_dir = logger.run_dir
    logger.log_event(
        "script_started",
        job_dir=os.path.dirname(os.path.realpath(__file__)),
        log_file=logger.log_file,
    )
    logger.log_info_block("ARGPARSE PARAMETERS", args)

    set_random_seed(args.seed)

    logger.log_event("model_initializing", model_arch=args.model_arch)
    model = build_model(args.model_arch, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    logger.log_info_block(
        "CHECKPOINT PARAMETERS",
        {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch", ""),
            "checkpoint_model_arch": checkpoint.get("model_arch", ""),
            "checkpoint_model_config": checkpoint.get("model_config", ""),
        },
    )

    wrapped_model = ConditionalVelocityModel(model).to(device)
    solver = ODESolver(velocity_model=wrapped_model)
    patch_processor = TensorPatchProcessor()
    time_grid = torch.tensor([0.0, 1.0], device=device)

    sample_transform = None
    if args.clip_vmin is not None or args.clip_vmax is not None:
        sample_transform = ClipFirstChannel(
            vmin=args.clip_vmin,
            vmax=args.clip_vmax,
        )
    dataset = SegyDataset(args.segy, transform=sample_transform)
    if args.shot_index < 0 or args.shot_index >= len(dataset):
        raise IndexError(
            f"--shot_index must be in [0, {len(dataset) - 1}], got {args.shot_index}."
        )
    logger.log_event(
        "dataset_initialized",
        segy=args.segy,
        num_shots=len(dataset),
    )

    logger.log_event("sampling_started", shot_index=args.shot_index, segy=args.segy)
    sample = dataset[args.shot_index][0].unsqueeze(0).unsqueeze(0).to(device)
    result = sample_one_shot(
        sample=sample,
        solver=solver,
        patch_processor=patch_processor,
        time_grid=time_grid,
        args=args,
        logger=logger,
    )
    shot_dir, psnr, mae = save_shot_outputs(
        result,
        args.shot_index,
        output_dir / f"shot_{args.shot_index:04d}",
    )
    logger.log_event(
        "sampling_finished",
        shot_index=args.shot_index,
        mask_ratio=result["mask_ratio"],
        psnr=psnr,
        mae=mae,
        saved=str(shot_dir),
    )

    logger.log_event("validation_finished", shot_index=args.shot_index, output_dir=str(output_dir))
    logger.close()


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
