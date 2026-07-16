import argparse
from pathlib import Path

import numpy as np
import torch

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.wrapper import DiTTransformer2DWrapper


class DimVelocityModel(ModelWrapper):
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
        description="Generate seismic image patches from dimension patches with a train6 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dim_patches", required=True, help="Input dimension patch NPY file [N,C,H,W].")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory produced by train6.py.")
    parser.add_argument("--output_npy", required=True, help="Output reconstructed image patch NPY file.")
    parser.add_argument("--device", default="cuda", help="Sampling device.")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of patches sampled at once.")
    parser.add_argument("--solver_step_size", default=0.05, type=float, help="Euler solver step size.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed used for sampling noise.")
    return parser


def validate_args(args):
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"--checkpoint must be a train6.py checkpoint directory, got {checkpoint_dir}.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.solver_step_size <= 0:
        raise ValueError("--solver_step_size must be positive.")


def load_dim_patches(path):
    dim_patches = np.load(path)
    if dim_patches.ndim != 4:
        raise ValueError(f"Expected dim patches with shape [N,C,H,W], got {dim_patches.shape}.")
    if dim_patches.shape[0] == 0:
        raise ValueError("Input dim patch file contains no patches.")
    return dim_patches.astype(np.float32, copy=False)


def load_checkpoint(checkpoint, device):
    print(f"[valid6] Loading checkpoint: {checkpoint}", flush=True)
    model, epoch, _ = DiTTransformer2DWrapper.from_training(
        save_directory=checkpoint,
        device=device,
    )
    model.eval()
    print(f"[valid6] Loaded checkpoint epoch={epoch}", flush=True)
    return model


def validate_model_channels(model, dim_patches):
    model_in_channels = int(model.model.config.in_channels)
    dim_channels = int(dim_patches.shape[1])
    expected_in_channels = 1 + dim_channels
    if model_in_channels != expected_in_channels:
        raise ValueError(
            "Checkpoint input channel count does not match dim patch data. "
            f"Checkpoint expects in_channels={model_in_channels}, but valid6 will pass "
            f"1 image/noise channel + {dim_channels} dimension channels = "
            f"{expected_in_channels} channels."
        )


def sample_patches(model, dim_patches, args, device):
    validate_model_channels(model, dim_patches)

    conditioning = torch.from_numpy(dim_patches).float().to(device)
    num_patches, _, patch_height, patch_width = conditioning.shape
    time_grid = torch.tensor([0.0, 1.0], device=device)
    solver = ODESolver(velocity_model=DimVelocityModel(model).to(device))

    batches = []
    num_batches = (num_patches + args.batch_size - 1) // args.batch_size
    print(
        f"[valid6] Sampling {num_patches} patches in {num_batches} batches.",
        flush=True,
    )

    for batch_index, start in enumerate(range(0, num_patches, args.batch_size), start=1):
        end = min(start + args.batch_size, num_patches)
        conditioning_batch = conditioning[start:end]
        noise = torch.randn(
            (end - start, 1, patch_height, patch_width),
            device=device,
            dtype=conditioning_batch.dtype,
        )
        print(f"[valid6] Sampling batch {batch_index}/{num_batches}: patches [{start}, {end})", flush=True)
        sampled = solver.sample(
            time_grid=time_grid,
            x_init=noise,
            return_intermediates=False,
            step_size=args.solver_step_size,
            cfg_scale=0.0,
            label=None,
            concat_conditioning={"concat_conditioning": conditioning_batch},
        )
        batches.append(sampled.detach().cpu())

    patches = torch.cat(batches, dim=0).squeeze(1).numpy()
    return patches


def main(args):
    validate_args(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim_patches = load_dim_patches(args.dim_patches)
    print(f"[valid6] Loaded dim patches: shape={dim_patches.shape}", flush=True)
    model = load_checkpoint(args.checkpoint, device)
    reconstructed_patches = sample_patches(model, dim_patches, args, device)

    output_npy = Path(args.output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, reconstructed_patches)
    print(
        f"[valid6] Saved reconstructed patches to {output_npy} "
        f"with shape={reconstructed_patches.shape}, "
        f"min={float(reconstructed_patches.min()):.6g}, "
        f"max={float(reconstructed_patches.max()):.6g}",
        flush=True,
    )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
