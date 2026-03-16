import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from build_patch_dataset import build_sample_transform
from core.dataset import SegyDataset
from core.logging.logger import SimpleLogger
from core.masks.row_mask import generate_random_row_mask
from core.metrics import compute_psnr
from core.patching import TensorPatchProcessor
from core.training import set_random_seed
from core.visualization import plot_seismic_grid
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from train4 import MODEL_CONFIGS, build_model


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
    parser = argparse.ArgumentParser(description="Validate a full seismic shot with a train4.py checkpoint.", )
    parser.add_argument("--segy", required=True, help="SEG-Y file used for sampling and reconstruction.", )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train4.py.", )
    parser.add_argument("--output_dir", default="./valid4_output",
                        help="Directory used to save figures and NPZ files.", )
    parser.add_argument("--device", default="cuda", help="Sampling device.", )
    parser.add_argument("--model_name", choices=sorted(MODEL_CONFIGS.keys()), default="dit",
                        help="Model architecture used during training.", )
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Fixed random row-mask ratio.", )
    parser.add_argument("--patch_size", default=32, type=int, help="Patch size used to train the patch dataset.", )
    parser.add_argument("--overlap_size", default=16, type=int, help="Overlap size used when extracting patches.", )
    parser.add_argument("--shot_index", default=0, type=int, help="Shot index to sample.", )
    parser.add_argument("--slice", nargs=2, type=int, default=[0, 1501],
                        help="Slice range on the last dimension. Use 0 0 to disable.", )
    parser.add_argument("--resize", nargs=2, type=int, default=[0, 0],
                        help="Resize target height width. Use 0 0 to disable.", )
    parser.add_argument("--solver_step_size", default=0.01, type=float, help="Euler solver step size.", )
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed used for mask generation and sampling noise.", )
    return parser


def normalize_patch_batch(
        patches: torch.Tensor,
        eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_mins = patches.amin(dim=(2, 3), keepdim=True)
    patch_maxs = patches.amax(dim=(2, 3), keepdim=True)
    normalized = 2.0 * (patches - patch_mins) / (patch_maxs - patch_mins + eps) - 1.0
    return normalized, patch_mins, patch_maxs


def denormalize_patch_batch(
        patches: torch.Tensor,
        patch_mins: torch.Tensor,
        patch_maxs: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (patches + 1.0) * (patch_maxs - patch_mins) + patch_mins


def validate_args(args):
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.overlap_size < 0:
        raise ValueError("--overlap_size must be non-negative.")
    if args.overlap_size >= args.patch_size:
        raise ValueError("--overlap_size must be smaller than --patch_size.")
    if not (0.0 <= args.mask_ratio < 1.0):
        raise ValueError("--mask_ratio must satisfy 0 <= value < 1.")

    model_config = MODEL_CONFIGS[args.model_name]
    expected_input_size = model_config.get("input_size")
    if expected_input_size is not None and args.patch_size != expected_input_size:
        raise ValueError(
            f"{args.model_name} expects patch_size={expected_input_size}, "
            f"but got {args.patch_size}."
        )


def sample_one_shot(
        sample: torch.Tensor,
        solver: ODESolver,
        patch_processor: TensorPatchProcessor,
        time_grid: torch.Tensor,
        args,
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

    normalized_clean_patches, patch_mins, patch_maxs = normalize_patch_batch(clean_patches)
    normalized_missed_patches = normalized_clean_patches * mask_patches

    sampled_patches = solver.sample(
        time_grid=time_grid,
        x_init=torch.randn_like(normalized_clean_patches),
        return_intermediates=False,
        step_size=args.solver_step_size,
        cfg_scale=0.0,
        label=None,
        concat_conditioning={
            "concat_conditioning": torch.cat(
                (normalized_missed_patches, mask_patches),
                dim=1,
            )
        },
    )

    reconstructed_normalized_patches = (
            normalized_missed_patches + (1.0 - mask_patches) * sampled_patches
    )

    # Use clean patch statistics to map predictions back for evaluation/visualization.
    reconstructed_patches = denormalize_patch_batch(
        reconstructed_normalized_patches,
        patch_mins,
        patch_maxs,
    ).unsqueeze(0)

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
    recon = result["recon"][0, 0].detach().cpu().clamp(-1.0, 1.0).numpy()
    diff = raw - recon

    np.savez_compressed(
        shot_dir / "result.npz",
        raw=raw,
        mask=mask,
        missed=missed,
        recon=recon,
        diff=diff,
        mask_ratio=np.array(result["mask_ratio"], dtype=np.float32),
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
    validate_args(args)
    shot_output_dir = Path(args.output_dir) / f"shot_{args.shot_index:04d}"
    shot_output_dir.mkdir(parents=True, exist_ok=True)

    logger = SimpleLogger(log_dir=str(shot_output_dir), overwrite=True)
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    set_random_seed(args.seed)

    model = build_model(args.model_name, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    wrapped_model = ConditionalVelocityModel(model).to(device)
    solver = ODESolver(velocity_model=wrapped_model)
    patch_processor = TensorPatchProcessor()
    time_grid = torch.tensor([0.0, 1.0], device=device)

    sample_transform = build_sample_transform(args)
    if sample_transform is None:
        sample_transform = transforms.Lambda(lambda x: x)

    dataset = SegyDataset(args.segy, transform=sample_transform)
    if args.shot_index < 0 or args.shot_index >= len(dataset):
        raise IndexError(
            f"--shot_index must be in [0, {len(dataset) - 1}], got {args.shot_index}."
        )

    logger.info(f"Sampling shot {args.shot_index} from {args.segy}")

    sample = dataset[args.shot_index][0].unsqueeze(0).unsqueeze(0).to(device)
    result = sample_one_shot(
        sample=sample,
        solver=solver,
        patch_processor=patch_processor,
        time_grid=time_grid,
        args=args,
    )
    shot_dir, psnr, mae = save_shot_outputs(result, args.shot_index, shot_output_dir)
    logger.info(
        f"shot={args.shot_index}, mask_ratio={result['mask_ratio']:.4f}, "
        f"psnr={psnr:.4f}, mae={mae:.6f}, saved={shot_dir}"
    )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
