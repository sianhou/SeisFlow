import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BuildStats:
    total_shots: int = 0
    saved_files: int = 0
    skipped_files: int = 0
    total_patches_before_filter: int = 0
    total_patches_saved: int = 0
    skipped_zero_patches: int = 0


def build_sample_transform(args):
    from torchvision import transforms

    from core.transforms import SliceLastDimension

    transform_list = []

    if 0 <= args.slice[0] < args.slice[1]:
        transform_list.append(
            SliceLastDimension(args.slice[0], args.slice[1])
        )

    if args.resize[0] > 0 and args.resize[1] > 0:
        transform_list.append(
            transforms.Resize((args.resize[0], args.resize[1]))
        )

    if not transform_list:
        return None

    return transforms.Compose(transform_list)


def filter_zero_patches(patches, positions):
    if patches.ndim != 3:
        raise ValueError(f"Expected patches to be [N,H,W], got {patches.shape}")

    keep_mask = np.any(patches != 0, axis=(1, 2))
    filtered_patches = patches[keep_mask]
    filtered_positions = [pos for pos, keep in zip(positions, keep_mask) if keep]
    skipped = int((~keep_mask).sum())
    return filtered_patches, filtered_positions, skipped


def normalize_patches_per_channel_abs(patches):
    import torch

    from core.transforms import AbsNormalize

    if patches.ndim != 3:
        raise ValueError(f"Expected patches to be [N,H,W], got {patches.shape}")

    patches_tensor = torch.from_numpy(patches).float().unsqueeze(1)  # [N,1,H,W]
    normalizer = AbsNormalize(per_channel=True)
    normalized = normalizer(patches_tensor)
    return normalized.squeeze(1).cpu().numpy()


def clip_patches(patches, vmin=None, vmax=None):
    import torch

    from core.transforms import Clip

    if patches.ndim != 3:
        raise ValueError(f"Expected patches to be [N,H,W], got {patches.shape}")
    if vmin is None and vmax is None:
        return patches

    patches_tensor = torch.from_numpy(patches).float().unsqueeze(1)  # [N,1,H,W]
    clipper = Clip(vmin=vmin, vmax=vmax, per_channel=True)
    clipped = clipper(patches_tensor)
    return clipped.squeeze(1).cpu().numpy()


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build single-channel seismic patch datasets for WP1. Supports slicing, "
            "resizing, and custom square patch sizes such as 128, 256, and 512."
        ),
        epilog=(
            "Examples:\n"
            "  Build 256x256 patches:\n"
            "    python scripts/build_patch_dataset.py --segy ma2+GathAP.sgy "
            "--patch_size 256 --overlap_size 16 --slice 0 1501 "
            "--resize 512 512 --normalize --output_dir ./train_dataset256"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--segy",
                        help="Input SEG-Y file used to build the patch dataset.")
    parser.add_argument(
        "--patch_size",
        default=256,
        type=int,
        help="Square patch size. WP1 commonly uses 128, 256, or 512.",
    )
    parser.add_argument("--overlap_size", default=16, type=int)
    parser.add_argument("--output_dir", default="./dataset_train")
    parser.add_argument("--clip_vmin", default=None, type=float,
                        help="Optional lower clipping bound applied before normalization.")
    parser.add_argument("--clip_vmax", default=None, type=float,
                        help="Optional upper clipping bound applied before normalization.")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize each saved patch independently with per_channel+abs to [-1, 1].")
    parser.add_argument("--slice", nargs=2, type=int, default=[0, 0],
                        help="Slice range on the last dimension as two ints: start end. Use 0 0 to disable.")
    parser.add_argument("--resize", nargs=2, type=int, default=[0, 0],
                        help="Resize target as two ints: height width. Use 0 0 to disable.")
    parser.add_argument("--plot_start", default=0, type=int,
                        help="First patch index used for visualization in test_data.")
    parser.add_argument("--plot_interval", default=100, type=int,
                        help="Plot every N patches from --plot_start in test_data.")
    return parser


def validate_args(args):
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.overlap_size < 0 or args.overlap_size >= args.patch_size:
        raise ValueError("--overlap_size must be in [0, patch_size).")
    if args.slice[0] < 0 or args.slice[1] < 0:
        raise ValueError("--slice values must be non-negative. Use 0 0 to disable.")
    if args.resize[0] < 0 or args.resize[1] < 0:
        raise ValueError("--resize values must be non-negative. Use 0 0 to disable.")
    if args.clip_vmin is not None and args.clip_vmax is not None and args.clip_vmin > args.clip_vmax:
        raise ValueError("--clip_vmin must be less than or equal to --clip_vmax.")
    if args.plot_start < 0:
        raise ValueError("--plot_start must be non-negative.")
    if args.plot_interval <= 0:
        raise ValueError("--plot_interval must be positive.")


def build_dataset(args):
    from core.dataset import SegyDataset
    from core.patching import NumpyPatchProcessor

    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    sample_transform = build_sample_transform(args)
    dataset = SegyDataset(args.segy, transform=sample_transform)

    patch_size = (args.patch_size, args.patch_size)
    overlap_size = (args.overlap_size, args.overlap_size)
    patch_processor = NumpyPatchProcessor()
    stats = BuildStats(total_shots=len(dataset))

    for i in range(len(dataset)):
        shot = dataset[i][0].numpy()
        patches, positions, original_shape = patch_processor.extract_overlapping_patches_2d(
            shot,
            patch_size=patch_size,
            overlap=overlap_size,
        )
        stats.total_patches_before_filter += len(patches)

        patches, positions, skipped_zero_patches = filter_zero_patches(patches, positions)
        stats.skipped_zero_patches += skipped_zero_patches

        if len(patches) == 0:
            stats.skipped_files += 1
            print(f"Skipped shot {i:04d}: all patches are zero")
            continue

        if args.clip_vmin is not None or args.clip_vmax is not None:
            patches = clip_patches(patches, vmin=args.clip_vmin, vmax=args.clip_vmax)

        if args.normalize:
            patches = normalize_patches_per_channel_abs(patches)

        output_file = os.path.join(args.output_dir, f"patches_{i:04d}.npy")
        np.save(output_file, patches)

        stats.saved_files += 1
        stats.total_patches_saved += len(patches)

        print(
            f"Saved {output_file} with {len(patches)} patches "
            f"(skipped zero patches: {skipped_zero_patches})"
        )

    print("\nBuild summary")
    print(f"segy: {args.segy}")
    print(f"output_dir: {args.output_dir}")
    print(f"total_shots: {stats.total_shots}")
    print(f"saved_files: {stats.saved_files}")
    print(f"skipped_files: {stats.skipped_files}")
    print(f"total_patches_before_filter: {stats.total_patches_before_filter}")
    print(f"total_patches_saved: {stats.total_patches_saved}")
    print(f"skipped_zero_patches: {stats.skipped_zero_patches}")
    print(f"slice_range: {args.slice}")
    print(f"resize: {args.resize}")
    print(f"clip_vmin: {args.clip_vmin}")
    print(f"clip_vmax: {args.clip_vmax}")
    print(f"normalize: {args.normalize}")


def test_data(args):
    import matplotlib.pyplot as plt

    from core.dataset import PatchDataset

    print("Creating patch dataset")
    pd = PatchDataset(data_path=args.output_dir)
    total_patches = len(pd)
    print(f"total patches: {total_patches}")

    if total_patches == 0:
        print("No patches available for visualization")
        return

    plot_start = min(args.plot_start, total_patches - 1)
    plot_indices = range(plot_start, total_patches, args.plot_interval)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for patch_index in plot_indices:
        patch = pd[patch_index]
        patch_np = patch.squeeze(0).detach().cpu().numpy()

        print(f"Visualizing patch index: {patch_index}")
        print(
            f"patch stats: min={patch_np.min():.6f}, "
            f"max={patch_np.max():.6f}, mean={patch_np.mean():.6f}"
        )

        plt.figure(figsize=(6, 6))
        if args.normalize:
            plt.imshow(patch_np.T, cmap="seismic", origin="upper", vmin=-1, vmax=1)
        else:
            plt.imshow(patch_np.T, cmap="seismic", origin="upper")
        plt.title(f"Patch #{patch_index}")
        plt.colorbar()
        plt.tight_layout()
        figure_file = os.path.join(figures_dir, f"patch_{patch_index:06d}.png")
        plt.savefig(figure_file, dpi=150)
        plt.close()
        print(f"Saved figure: {figure_file}")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    build_dataset(args)
    test_data(args)
