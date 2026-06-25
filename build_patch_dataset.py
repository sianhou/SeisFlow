import argparse
import os

import numpy as np


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


def build_pos_arrays(shape):
    height, width = shape
    pos_trace = np.arange(height, dtype=np.float32)[:, None]
    pos_sample = np.arange(width, dtype=np.float32)[None, :]
    return (
        np.broadcast_to(pos_trace, (height, width)).copy(),
        np.broadcast_to(pos_sample, (height, width)).copy(),
    )


def build_auxiliary_patches(sample, patch_processor, patch_size, overlap, positions, original_shape):
    sx = sample[1].numpy()
    sy = sample[2].numpy()
    rx = sample[3].numpy()
    ry = sample[4].numpy()
    pos_trace, pos_sample = build_pos_arrays(original_shape)

    sx_patches, sx_positions, sx_shape = patch_processor.extract_overlapping_patches_2d(
        sx,
        patch_size=patch_size,
        overlap=overlap,
    )
    sy_patches, sy_positions, sy_shape = patch_processor.extract_overlapping_patches_2d(
        sy,
        patch_size=patch_size,
        overlap=overlap,
    )
    rx_patches, rx_positions, rx_shape = patch_processor.extract_overlapping_patches_2d(
        rx,
        patch_size=patch_size,
        overlap=overlap,
    )
    ry_patches, ry_positions, ry_shape = patch_processor.extract_overlapping_patches_2d(
        ry,
        patch_size=patch_size,
        overlap=overlap,
    )
    pos_trace_patches, pos_trace_positions, pos_trace_shape = patch_processor.extract_overlapping_patches_2d(
        pos_trace,
        patch_size=patch_size,
        overlap=overlap,
    )
    pos_sample_patches, pos_sample_positions, pos_sample_shape = patch_processor.extract_overlapping_patches_2d(
        pos_sample,
        patch_size=patch_size,
        overlap=overlap,
    )

    for name, channel_positions, channel_shape in (
            ("sx", sx_positions, sx_shape),
            ("sy", sy_positions, sy_shape),
            ("rx", rx_positions, rx_shape),
            ("ry", ry_positions, ry_shape),
            ("pos_trace", pos_trace_positions, pos_trace_shape),
            ("pos_sample", pos_sample_positions, pos_sample_shape),
    ):
        if not np.array_equal(positions, channel_positions) or original_shape != channel_shape:
            raise RuntimeError(f"{name} patch grid does not match shot patch grid.")

    auxiliary_patches = np.stack(
        [sx_patches, sy_patches, rx_patches, ry_patches, pos_trace_patches, pos_sample_patches],
        axis=1,
    )
    return auxiliary_patches


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


def build_aux_output_dir(output_dir):
    return f"{output_dir.rstrip(os.sep)}_aux"


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
    parser.add_argument("--auxiliary_data", action="store_true",
                        help="Save 6-channel auxiliary patches [sx, sy, rx, ry, pos_trace, pos_sample] instead of amplitude patches.")
    parser.add_argument("--keep_zeros_patch", action="store_true",
                        help="Keep patches whose amplitude channel is all zero. By default, all-zero amplitude patches are skipped.")
    parser.add_argument("--slice", nargs=2, type=int, default=[0, 0],
                        help="Slice range on the last dimension as two ints: start end. Use 0 0 to disable.")
    parser.add_argument("--resize", nargs=2, type=int, default=[0, 0],
                        help="Resize target as two ints: height width. Use 0 0 to disable.")
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


def build_dataset(args):
    from core.dataset import SegyDataset
    from core.patching import NumpyPatchProcessor

    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    aux_output_dir = build_aux_output_dir(args.output_dir)
    if args.auxiliary_data:
        os.makedirs(aux_output_dir, exist_ok=True)

    sample_transform = build_sample_transform(args)
    dataset = SegyDataset(args.segy, transform=sample_transform)

    patch_size = (args.patch_size, args.patch_size)
    overlap_size = (args.overlap_size, args.overlap_size)
    patch_processor = NumpyPatchProcessor()

    for i in range(len(dataset)):
        sample = dataset[i]
        shot = sample[0].numpy()
        patches, positions, original_shape = patch_processor.extract_overlapping_patches_2d(
            shot,
            patch_size=patch_size,
            overlap=overlap_size,
        )

        if args.auxiliary_data:
            auxiliary_patches = build_auxiliary_patches(
                sample,
                patch_processor=patch_processor,
                patch_size=patch_size,
                overlap=overlap_size,
                positions=positions,
                original_shape=original_shape,
            )

        if args.keep_zeros_patch:
            skipped_zero_patches = 0
        else:
            keep_mask = np.any(patches != 0, axis=(1, 2))
            patches = patches[keep_mask]
            positions = [pos for pos, keep in zip(positions, keep_mask) if keep]
            if args.auxiliary_data:
                auxiliary_patches = auxiliary_patches[keep_mask]
            skipped_zero_patches = int((~keep_mask).sum())

        if len(patches) == 0:
            print(f"Skipped shot {i:04d}: all patches are zero")
            continue

        if args.clip_vmin is not None or args.clip_vmax is not None:
            patches = clip_patches(patches, vmin=args.clip_vmin, vmax=args.clip_vmax)

        if args.normalize:
            patches = normalize_patches_per_channel_abs(patches)

        output_file = os.path.join(args.output_dir, f"patches_{i:04d}.npy")
        np.save(output_file, patches)

        if args.auxiliary_data:
            aux_output_file = os.path.join(aux_output_dir, f"patches_{i:04d}.npy")
            np.save(aux_output_file, auxiliary_patches)

        print(
            f"Saved {output_file} with {len(patches)} patches "
            f"(skipped zero patches: {skipped_zero_patches})"
        )


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    build_dataset(args)
