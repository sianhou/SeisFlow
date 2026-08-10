import argparse
import os
import tempfile

import numpy as np


class ArgumentFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build shot-level seismic image patches, coordinate-condition patches, "
            "and reconstruction metadata from a SEG-Y file."
        ),
        epilog=(
            "Examples:\n"
            "  Build normalized 256x256 training patches:\n"
            "    python build_shot_dataset.py --segy ma2+GathAP.sgy "
            "--patch_size 256 --overlap_size 16 --slice 0 1501 "
            "--resize 512 512 --clip -2 2 --normalize "
            "--output_dir ./dataset\n"
            "\n"
            "  Build train/valid splits with grouped random validation shots:\n"
            "    python build_shot_dataset.py --segy ma2+GathAP.sgy "
            "--valid 0.2 --valid_mode group_random --seed 42 "
            "--output_dir ./dataset"
        ),
        formatter_class=ArgumentFormatter,
    )
    parser.add_argument("--segy", required=True, default=argparse.SUPPRESS,
                        help="Input SEG-Y file.")
    parser.add_argument(
        "--patch_size",
        default=256,
        type=int,
        help="Height and width of each square patch.",
    )
    parser.add_argument("--overlap_size", default=16, type=int,
                        help="Number of pixels shared by neighboring patches along both axes.")
    parser.add_argument("--output_dir", default="./dataset",
                        help=(
                            "Output root. The script writes train, train_dim, train_aux, "
                            "and, when --valid > 0, valid, valid_dim, valid_aux."
                        ))
    parser.add_argument("--valid", default=0.0, type=float,
                        help="Validation split ratio over shots. Must satisfy 0 <= value < 1.")
    parser.add_argument(
        "--valid_mode",
        default="uniform",
        choices=[
            "uniform",
            "random",
            "group_random",
        ],
        help=(
            "How validation shots are selected: uniform selects evenly spaced shots; "
            "random samples validation shots globally; group_random splits all shots "
            "into uniform groups and samples one shot per group."
        ),
    )
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed used by random and group_random validation modes.")
    parser.add_argument("--clip", nargs=2, default=None, type=float, metavar=("VMIN", "VMAX"),
                        help="Clip image-patch amplitudes to [VMIN, VMAX] before normalization.")
    parser.add_argument("--normalize", action="store_true",
                        help=(
                            "Normalize each image patch by its maximum absolute amplitude. "
                            "The per-patch scale is saved in the aux metadata."
                        ))
    parser.add_argument("--keep_zeros_patch", action="store_true",
                        help="Keep all-zero image patches; by default they are skipped.")
    parser.add_argument("--slice", nargs=2, type=int, default=[0, 0], metavar=("START", "END"),
                        help="Slice the sample/time axis as [START, END) before patch extraction. Use 0 0 to disable.")
    parser.add_argument("--resize", nargs=2, type=int, default=[0, 0], metavar=("HEIGHT", "WIDTH"),
                        help="Resize each shot to HEIGHT x WIDTH before patch extraction. Use 0 0 to disable.")
    parser.add_argument("--no_shot_plot", action="store_true",
                        help="Do not save the default generated train/valid shot plot under output_dir.")
    return parser


def validate_args(args):
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.overlap_size < 0 or args.overlap_size >= args.patch_size:
        raise ValueError("--overlap_size must be in [0, patch_size).")
    if args.slice[0] < 0 or args.slice[1] < 0:
        raise ValueError("--slice values must be non-negative. Use 0 0 to disable.")
    if args.slice != [0, 0] and args.slice[0] >= args.slice[1]:
        raise ValueError("--slice must be 0 0 or satisfy START < END.")
    if args.resize[0] < 0 or args.resize[1] < 0:
        raise ValueError("--resize values must be non-negative. Use 0 0 to disable.")
    if (args.resize[0] == 0) != (args.resize[1] == 0):
        raise ValueError("--resize must set both HEIGHT and WIDTH, or use 0 0 to disable.")
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
        f"{output_prefix}/train_dim",
        f"{output_prefix}/train_aux",
        f"{output_prefix}/valid",
        f"{output_prefix}/valid_dim",
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


def plot_shot_presence(present_shot_keys, missing_shot_keys, output_file):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), "seisflow_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    present_numbers = np.unique(np.asarray(present_shot_keys, dtype=np.int64))
    missing_numbers = np.unique(np.asarray(missing_shot_keys, dtype=np.int64))
    shot_numbers = np.unique(np.concatenate([present_numbers, missing_numbers]))
    if shot_numbers.size == 0:
        raise ValueError("Cannot plot shot presence without any generated shots.")

    first_shot = int(shot_numbers.min())
    last_shot = int(shot_numbers.max())
    x = np.arange(first_shot, last_shot + 1, dtype=np.int64)
    present_x = np.intersect1d(x, present_numbers, assume_unique=True)
    missing_x = np.intersect1d(x, missing_numbers, assume_unique=True)
    present_count = int(present_x.size)
    missing_count = int(missing_x.size)
    total_count = present_count + missing_count
    missing_ratio = missing_count / total_count if total_count else 0.0

    fig_width = min(18.0, max(10.0, total_count / 80.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhspan(-0.12, 1.12, color="#F1F3F5", zorder=0)

    if present_x.size:
        ax.vlines(
            present_x,
            0.0,
            1.0,
            color="#D32F2F",
            alpha=0.45,
            linewidth=0.9,
            zorder=2,
        )
        ax.scatter(
            present_x,
            np.ones_like(present_x),
            s=28,
            facecolors="white",
            edgecolors="#D32F2F",
            linewidths=1.2,
            label=f"train / present ({present_count})",
            zorder=3,
        )

    if missing_x.size:
        ax.scatter(
            missing_x,
            np.zeros_like(missing_x),
            s=30,
            color="#1976D2",
            marker="x",
            linewidths=1.2,
            label=f"valid / missing ({missing_count})",
            zorder=4,
        )

    ax.set_xlabel("shot number")
    ax.set_ylabel("status")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["missing", "present"])
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlim(first_shot - 0.5, last_shot + 0.5)
    ax.set_title("Generated Train/Valid Shot Distribution")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.grid(True, axis="x", color="#B0BEC5", alpha=0.35, linewidth=0.8)
    ax.grid(True, axis="y", color="#CFD8DC", alpha=0.75, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#90A4AE")
    ax.spines["bottom"].set_color("#90A4AE")
    ax.tick_params(colors="#37474F")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CFD8DC")
    ax.text(
        0.01,
        0.96,
        (
            f"range: {first_shot}..{last_shot}\n"
            f"train/present: {present_count} / {total_count}\n"
            f"valid/missing: {missing_count} ({missing_ratio:.1%})"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#263238",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CFD8DC",
            "alpha": 0.95,
        },
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    return {
        "first_shot": first_shot,
        "last_shot": last_shot,
        "present_count": present_count,
        "missing_count": missing_count,
        "total_count": total_count,
    }


def select_uniform_indices(num_items, count):
    return np.floor((np.arange(count) + 0.5) * num_items / count).astype(int)


def build_split_indices(num_shots, valid_ratio, valid_mode, seed=0):
    if valid_mode not in {"uniform", "random", "group_random"}:
        raise ValueError(f"Unsupported valid_mode: {valid_mode}")
    if valid_ratio <= 0:
        return set(range(num_shots)), set()
    if num_shots < 2:
        raise ValueError("--valid requires at least 2 shots.")

    valid_count = int(round(num_shots * valid_ratio))
    valid_count = max(1, min(valid_count, num_shots - 1))

    if valid_mode == "uniform":
        valid_indices = select_uniform_indices(num_shots, valid_count)
    elif valid_mode == "random":
        rng = np.random.default_rng(seed)
        valid_indices = rng.choice(num_shots, size=valid_count, replace=False)
    else:
        rng = np.random.default_rng(seed)
        groups = np.array_split(np.arange(num_shots), valid_count)
        valid_indices = np.asarray(
            [rng.choice(group) for group in groups if len(group) > 0],
            dtype=np.int64,
        )

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
    train_indices, valid_indices = build_split_indices(
        len(dataset),
        args.valid,
        args.valid_mode,
        seed=args.seed,
    )
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
    train_shot_numbers = []
    valid_shot_numbers = []

    for i in range(len(dataset)):
        sample = dataset[i]
        shot_number = dataset.shot_keys[i]
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

        if split_name == "valid":
            valid_shot_numbers.append(shot_number)
        else:
            train_shot_numbers.append(shot_number)

        print(
            f"Saved {split_name} {output_file} with {len(patches)} patches "
            f"(skipped zero patches: {skipped_zero_patches})"
        )

    if not args.no_shot_plot:
        shot_plot_file = os.path.join(args.output_dir.rstrip(os.sep), "shot_presence.png")
        shot_plot_stats = plot_shot_presence(
            train_shot_numbers,
            valid_shot_numbers,
            shot_plot_file,
        )
        print(
            f"Saved shot presence plot {shot_plot_file} "
            f"(train/present={shot_plot_stats['present_count']}, "
            f"valid/missing={shot_plot_stats['missing_count']}, "
            f"range={shot_plot_stats['first_shot']}..{shot_plot_stats['last_shot']})"
        )


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    build_dataset(args)
