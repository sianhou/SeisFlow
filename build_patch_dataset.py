import argparse
import os

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build seismic image and dimension-coordinate patch datasets from SEG-Y shots.",
        epilog=(
            "Examples:\n"
            "  Build normalized 256x256 image and dimension patches:\n"
            "    python build_patch_dataset.py --segy ma2+GathAP.sgy "
            "--patch_size 256 --overlap_size 16 --slice 0 1501 "
            "--resize 512 512 --clip -2 2 --normalize "
            "--output_dir ./dataset"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--segy",
                        help="Input SEG-Y file.")
    parser.add_argument(
        "--patch_size",
        default=256,
        type=int,
        help="Square patch size.",
    )
    parser.add_argument("--overlap_size", default=16, type=int,
                        help="Overlap size between neighboring patches.")
    parser.add_argument("--output_dir", default="./dataset",
                        help="Output path prefix for train/valid image, dimension, and patch metadata directories.")
    parser.add_argument("--valid", default=0.0, type=float,
                        help="Validation split ratio by shot. Must be >= 0 and < 1.")
    parser.add_argument("--valid_mode", default="shot", choices=["shot"],
                        help="Validation split mode.")
    parser.add_argument("--clip", nargs=2, default=None, type=float, metavar=("VMIN", "VMAX"),
                        help="Amplitude clipping range applied before normalization.")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize each image patch by its own maximum absolute amplitude.")
    parser.add_argument("--keep_zeros_patch", action="store_true",
                        help="Keep all-zero image patches instead of skipping them.")
    parser.add_argument("--slice", nargs=2, type=int, default=[0, 0],
                        help="Sample-axis slice range START END. Use 0 0 to disable.")
    parser.add_argument("--resize", nargs=2, type=int, default=[0, 0],
                        help="Resize each shot to HEIGHT WIDTH before patch extraction. Use 0 0 to disable.")
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
    if args.clip is not None and args.clip[0] > args.clip[1]:
        raise ValueError("--clip VMIN must be less than or equal to VMAX.")
    if args.valid < 0 or args.valid >= 1:
        raise ValueError("--valid must be >= 0 and < 1.")


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


def build_t_array(shape):
    height, width = shape
    t = np.arange(width, dtype=np.float32)[None, :]
    return np.broadcast_to(t, (height, width)).copy()


def normalize_coordinate(values, lower, upper):
    denominator = upper - lower
    if denominator == 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lower) / denominator).astype(np.float32)


def mark_padding_invalid(patches, positions, original_shape, invalid_value=-1.0):
    height, width = original_shape
    _, _, patch_height, patch_width = patches.shape

    for patch_index, (top, left) in enumerate(positions):
        bottom = min(top + patch_height, height)
        right = min(left + patch_width, width)
        valid_height = bottom - top
        valid_width = right - left

        if valid_height < patch_height:
            patches[patch_index, :, valid_height:, :] = invalid_value
        if valid_width < patch_width:
            patches[patch_index, :, :, valid_width:] = invalid_value

    return patches


def scan_dim_bounds(dataset):
    x0 = np.inf
    x1 = -np.inf
    y0 = np.inf
    y1 = -np.inf
    t0 = 0.0
    t1 = -np.inf

    for i in range(len(dataset)):
        sample = dataset[i]
        sx = sample[1].numpy()
        sy = sample[2].numpy()
        rx = sample[3].numpy()
        ry = sample[4].numpy()
        _, width = sample[0].shape

        x0 = min(x0, float(sx.min()), float(rx.min()))
        x1 = max(x1, float(sx.max()), float(rx.max()))
        y0 = min(y0, float(sy.min()), float(ry.min()))
        y1 = max(y1, float(sy.max()), float(ry.max()))
        t1 = max(t1, float(width - 1))

    if not np.isfinite([x0, x1, y0, y1, t0, t1]).all():
        raise RuntimeError("Failed to scan dimension coordinate bounds.")

    return x0, x1, y0, y1, t0, t1


def build_dim_patches(
        sample,
        patch_processor,
        patch_size,
        overlap,
        positions,
        original_shape,
        x0,
        x1,
        y0,
        y1,
        t0,
        t1,
):
    sx = sample[1].numpy()
    sy = sample[2].numpy()
    rx = sample[3].numpy()
    ry = sample[4].numpy()
    t = build_t_array(original_shape)

    sx = normalize_coordinate(sx, x0, x1)
    rx = normalize_coordinate(rx, x0, x1)
    sy = normalize_coordinate(sy, y0, y1)
    ry = normalize_coordinate(ry, y0, y1)
    t = normalize_coordinate(t, t0, t1)

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
    t_patches, t_positions, t_shape = patch_processor.extract_overlapping_patches_2d(
        t,
        patch_size=patch_size,
        overlap=overlap,
    )

    for name, channel_positions, channel_shape in (
            ("sx", sx_positions, sx_shape),
            ("sy", sy_positions, sy_shape),
            ("rx", rx_positions, rx_shape),
            ("ry", ry_positions, ry_shape),
            ("t", t_positions, t_shape),
    ):
        if not np.array_equal(positions, channel_positions) or original_shape != channel_shape:
            raise RuntimeError(f"{name} patch grid does not match shot patch grid.")

    dim_patches = np.stack(
        [sx_patches, sy_patches, rx_patches, ry_patches, t_patches],
        axis=1,
    )
    return mark_padding_invalid(dim_patches, positions, original_shape)


def normalize_patches_per_channel_abs(patches):
    import torch

    from core.transforms import AbsNormalize

    if patches.ndim != 3:
        raise ValueError(f"Expected patches to be [N,H,W], got {patches.shape}")

    patches_tensor = torch.from_numpy(patches).float().unsqueeze(1)  # [N,1,H,W]
    normalizer = AbsNormalize(per_channel=True)
    normalized, scales = normalizer.run(patches_tensor)
    return (
        normalized.squeeze(1).cpu().numpy(),
        scales[:, 0, 0, 0].cpu().numpy(),
    )


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


def build_output_dirs(output_dir):
    output_prefix = output_dir.rstrip(os.sep)
    return (
        f"{output_prefix}/train",
        f"{output_prefix}/dim_train",
        f"{output_prefix}/train_aux",
        f"{output_prefix}/valid",
        f"{output_prefix}/dim_valid",
        f"{output_prefix}/valid_aux",
    )


def save_patch_metadata(
        output_file,
        positions,
        original_shape,
        patch_scales,
):
    np.savez(
        output_file,
        positions=np.asarray(positions, dtype=np.int64),
        original_shape=np.asarray(original_shape, dtype=np.int64),
        patch_scales=np.asarray(patch_scales, dtype=np.float32),
    )


def build_split_indices(num_shots, valid_ratio, valid_mode):
    if valid_mode != "shot":
        raise ValueError(f"Unsupported valid_mode: {valid_mode}")
    if valid_ratio <= 0:
        return set(range(num_shots)), set()
    if num_shots < 2:
        raise ValueError("--valid requires at least 2 shots.")

    valid_count = int(round(num_shots * valid_ratio))
    valid_count = max(1, min(valid_count, num_shots - 1))
    valid_indices = np.floor((np.arange(valid_count) + 0.5) * num_shots / valid_count).astype(int)
    valid_indices = set(valid_indices.tolist())
    train_indices = set(range(num_shots)) - valid_indices
    return train_indices, valid_indices


def build_dataset(args):
    from core.dataset import SegyDataset
    from core.patching import NumpyPatchProcessor

    validate_args(args)
    (
        train_output_dir,
        train_dim_output_dir,
        train_aux_output_dir,
        valid_output_dir,
        valid_dim_output_dir,
        valid_aux_output_dir,
    ) = build_output_dirs(args.output_dir)
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(train_dim_output_dir, exist_ok=True)
    os.makedirs(train_aux_output_dir, exist_ok=True)
    if args.valid > 0:
        os.makedirs(valid_output_dir, exist_ok=True)
        os.makedirs(valid_dim_output_dir, exist_ok=True)
        os.makedirs(valid_aux_output_dir, exist_ok=True)

    sample_transform = build_sample_transform(args)
    dataset = SegyDataset(args.segy, transform=sample_transform)
    train_indices, valid_indices = build_split_indices(len(dataset), args.valid, args.valid_mode)
    if args.valid > 0:
        print(
            f"Split shots with mode={args.valid_mode}: "
            f"train={len(train_indices)}, valid={len(valid_indices)}"
        )
    dim_bounds = scan_dim_bounds(dataset)
    x0, x1, y0, y1, t0, t1 = dim_bounds
    print(
        "Dimension coordinate bounds: "
        f"x=({x0}, {x1}), y=({y0}, {y1}), t=({t0}, {t1})"
    )

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

        dim_patches = build_dim_patches(
            sample,
            patch_processor=patch_processor,
            patch_size=patch_size,
            overlap=overlap_size,
            positions=positions,
            original_shape=original_shape,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            t0=t0,
            t1=t1,
        )

        if args.keep_zeros_patch:
            skipped_zero_patches = 0
        else:
            keep_mask = np.any(patches != 0, axis=(1, 2))
            patches = patches[keep_mask]
            positions = positions[keep_mask]
            dim_patches = dim_patches[keep_mask]
            skipped_zero_patches = int((~keep_mask).sum())

        if len(patches) == 0:
            print(f"Skipped shot {i:04d}: all patches are zero")
            continue

        if args.clip is not None:
            patches = clip_patches(patches, vmin=args.clip[0], vmax=args.clip[1])

        patch_scales = np.ones((len(patches),), dtype=np.float32)
        if args.normalize:
            patches, patch_scales = normalize_patches_per_channel_abs(patches)

        if i in valid_indices:
            split_name = "valid"
            output_dir = valid_output_dir
            dim_output_dir = valid_dim_output_dir
            aux_output_dir = valid_aux_output_dir
        else:
            split_name = "train"
            output_dir = train_output_dir
            dim_output_dir = train_dim_output_dir
            aux_output_dir = train_aux_output_dir

        output_file = os.path.join(output_dir, f"patches_{i:04d}.npy")
        np.save(output_file, patches)

        dim_output_file = os.path.join(dim_output_dir, f"patches_{i:04d}.npy")
        np.save(dim_output_file, dim_patches)

        aux_output_file = os.path.join(aux_output_dir, f"patches_{i:04d}.npz")
        save_patch_metadata(
            aux_output_file,
            positions=positions,
            original_shape=original_shape,
            patch_scales=patch_scales,
        )

        print(
            f"Saved {split_name} {output_file} with {len(patches)} patches "
            f"(skipped zero patches: {skipped_zero_patches})"
        )


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    build_dataset(args)
