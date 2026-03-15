import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from core.dataset import SegyDataset, PatchDataset
from core.patching import extract_overlapping_patches_2d
from core.transforms import SliceLastDimension


@dataclass
class BuildStats:
    total_shots: int = 0
    saved_files: int = 0
    skipped_files: int = 0
    total_patches_before_filter: int = 0
    total_patches_saved: int = 0
    skipped_zero_patches: int = 0


def build_sample_transform(args):
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


def Normalize(patches, scale_range=(-1, 1), eps=1e-12):
    if patches.ndim != 3:
        raise ValueError(f"Expected patches to be [N,H,W], got {patches.shape}")

    low, high = scale_range
    if low >= high:
        raise ValueError("scale_range must satisfy min < max")

    patch_min = patches.min(axis=(1, 2), keepdims=True)
    patch_max = patches.max(axis=(1, 2), keepdims=True)
    normalized = 2.0 * (patches - patch_min) / (patch_max - patch_min + eps) - 1.0
    return (normalized + 1.0) * 0.5 * (high - low) + low


def create_parser():
    parser = argparse.ArgumentParser(description="Build patch dataset from SEG-Y shots")
    parser.add_argument("--segy",
                        help="Input SEG-Y file used to build the patch dataset.")
    parser.add_argument("--patch_size", default=32, type=int)
    parser.add_argument("--overlap_size", default=16, type=int)
    parser.add_argument("--output_dir", default="./dataset_train")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize each saved patch independently to [-1, 1].")
    parser.add_argument("--slice", nargs=2, type=int, default=[0, 1501],
                        help="Slice range on the last dimension as two ints: start end. Use 0 0 to disable.")
    parser.add_argument("--resize", nargs=2, type=int, default=[0, 0],
                        help="Resize target as two ints: height width. Use 0 0 to disable.")
    parser.add_argument("--test_index", default=0, type=int,
                        help="Patch index used for visualization in test_data.")
    return parser


def build_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)

    sample_transform = build_sample_transform(args)
    dataset = SegyDataset(args.segy, transform=sample_transform)

    patch_size = (args.patch_size, args.patch_size)
    overlap_size = (args.overlap_size, args.overlap_size)
    stats = BuildStats(total_shots=len(dataset))

    for i in range(len(dataset)):
        shot = dataset[i][0].numpy()
        patches, positions, original_shape = extract_overlapping_patches_2d(
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

        if args.normalize:
            patches = Normalize(patches)

        output_file = os.path.join(args.output_dir, f"patches_{i:04d}.npz")
        positions_array = np.array(positions)

        np.savez_compressed(
            output_file,
            patches=patches,
            positions=positions_array,
            original_shape=original_shape,
            patch_size=patch_size,
            overlap_size=overlap_size
        )

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
    print(f"normalize: {args.normalize}")


def test_data(args):
    print("Creating patch dataset")
    pd = PatchDataset(data_path=args.output_dir)
    print(f"total patches: {len(pd)}")

    test_index = max(0, min(args.test_index, len(pd) - 1))
    patch = pd[test_index]
    patch_np = patch.squeeze(0).detach().cpu().numpy()

    print(f"Visualizing patch index: {test_index}")
    print(
        f"patch stats: min={patch_np.min():.6f}, "
        f"max={patch_np.max():.6f}, mean={patch_np.mean():.6f}"
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(patch_np.T, cmap="seismic", origin="upper")
    plt.title(f"Patch #{test_index}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    build_dataset(args)
    test_data(args)
