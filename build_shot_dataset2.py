import argparse
import math
import os

import numpy as np


class ArgumentFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build six-channel seismic shot patches with global preprocessing. "
            "The six channels are seismic, SX, SY, RX, RY, and T."
        ),
        formatter_class=ArgumentFormatter,
    )
    parser.add_argument("--segy", required=True, help="Input SEG-Y file.")
    parser.add_argument(
        "--patch_size",
        default=256,
        type=int,
        help="Height and width of each square patch.",
    )
    parser.add_argument(
        "--overlap_size",
        default=16,
        type=int,
        help="Number of pixels shared by neighboring patches along both axes.",
    )
    parser.add_argument(
        "--output_dir",
        default="./dataset",
        help=(
            "Output root. Writes train, train_dim and, when --valid > 0, "
            "valid and valid_dim."
        ),
    )
    parser.add_argument(
        "--valid",
        default=0.0,
        type=float,
        help="Validation split ratio over shots. Must satisfy 0 <= value < 1.",
    )
    parser.add_argument(
        "--valid_mode",
        default="uniform",
        choices=["uniform", "random", "group_random"],
        help=(
            "How validation shots are selected: uniform selects evenly spaced shots; "
            "random samples validation shots globally; group_random samples one shot "
            "from each uniform group."
        ),
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for random validation-shot selection.",
    )
    parser.add_argument(
        "--clip",
        nargs=2,
        default=None,
        type=float,
        metavar=("VMIN", "VMAX"),
        help="Clip seismic amplitudes before slicing and global normalization.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalize only the seismic channel globally after clip/slice using one "
            "shared max-absolute scale, so zero remains exactly zero."
        ),
    )
    parser.add_argument(
        "--slice",
        nargs=2,
        type=int,
        default=[0, 0],
        metavar=("START", "END"),
        help="Slice the sample/time axis as [START, END). Use 0 0 to disable.",
    )
    return parser


def validate_args(args):
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.overlap_size < 0 or args.overlap_size >= args.patch_size:
        raise ValueError("--overlap_size must be in [0, patch_size).")
    if args.valid < 0 or args.valid >= 1:
        raise ValueError("--valid must be >= 0 and < 1.")
    if args.slice[0] < 0 or args.slice[1] < 0:
        raise ValueError("--slice values must be non-negative.")
    if args.slice != [0, 0] and args.slice[0] >= args.slice[1]:
        raise ValueError("--slice must be 0 0 or satisfy START < END.")
    if args.clip is not None and args.clip[0] > args.clip[1]:
        raise ValueError("--clip VMIN must be less than or equal to VMAX.")
def build_output_dirs(output_dir):
    prefix = output_dir.rstrip(os.sep)
    return {
        "train": f"{prefix}/train",
        "train_dim": f"{prefix}/train_dim",
        "train_aux": f"{prefix}/train_aux",
        "valid": f"{prefix}/valid",
        "valid_dim": f"{prefix}/valid_dim",
        "valid_aux": f"{prefix}/valid_aux",
    }


def build_split_indices(num_shots, valid_ratio, valid_mode, seed):
    if valid_ratio <= 0:
        return set(range(num_shots)), set()
    if num_shots < 2:
        raise ValueError("--valid requires at least 2 shots.")

    valid_count = int(round(num_shots * valid_ratio))
    valid_count = max(1, min(valid_count, num_shots - 1))

    if valid_mode == "uniform":
        valid_indices = np.floor(
            (np.arange(valid_count) + 0.5) * num_shots / valid_count
        ).astype(np.int64)
    elif valid_mode == "random":
        rng = np.random.default_rng(seed)
        valid_indices = rng.choice(
            num_shots, size=valid_count, replace=False
        )
    else:
        rng = np.random.default_rng(seed)
        groups = np.array_split(np.arange(num_shots), valid_count)
        valid_indices = np.asarray(
            [rng.choice(group) for group in groups if len(group)],
            dtype=np.int64,
        )

    valid_indices = set(valid_indices.tolist())
    return set(range(num_shots)) - valid_indices, valid_indices


def load_six_volumes(dataset, shot_index, slice_range, clip_range):
    """Load one shot as [6, H, W]: seismic, SX, SY, RX, RY, T."""
    sample = dataset[shot_index].numpy().astype(np.float32, copy=False)
    if sample.ndim != 3 or sample.shape[0] != 5:
        raise ValueError(
            f"Expected SegyDataset sample with shape [5, H, W], got {sample.shape}."
        )

    sample_start = 0
    if slice_range != [0, 0]:
        start, end = slice_range
        sample_start = start
        sample = sample[:, :, start:end]

    seismic = sample[0].copy()
    if clip_range is not None:
        seismic = np.clip(seismic, clip_range[0], clip_range[1])

    height, width = seismic.shape
    t = np.broadcast_to(
        np.arange(sample_start, sample_start + width, dtype=np.float32)[None, :],
        (height, width),
    ).copy()

    return np.stack(
        [seismic, sample[1], sample[2], sample[3], sample[4], t],
        axis=0,
    ).astype(np.float32, copy=False)


def scan_global_min_max(dataset, args):
    """Scan seismic extrema and grouped global coordinate extrema."""
    seismic_min = np.inf
    seismic_max = -np.inf
    coord_min = np.full(3, np.inf, dtype=np.float64)  # x, y, t
    coord_max = np.full(3, -np.inf, dtype=np.float64)

    for shot_index in range(len(dataset)):
        volumes = load_six_volumes(
            dataset,
            shot_index,
            args.slice,
            args.clip,
        )
        seismic_min = min(seismic_min, float(volumes[0].min()))
        seismic_max = max(seismic_max, float(volumes[0].max()))
        coord_min[0] = min(
            coord_min[0],
            float(volumes[1].min()),
            float(volumes[3].min()),
        )
        coord_max[0] = max(
            coord_max[0],
            float(volumes[1].max()),
            float(volumes[3].max()),
        )
        coord_min[1] = min(
            coord_min[1],
            float(volumes[2].min()),
            float(volumes[4].min()),
        )
        coord_max[1] = max(
            coord_max[1],
            float(volumes[2].max()),
            float(volumes[4].max()),
        )
        coord_min[2] = min(coord_min[2], float(volumes[5].min()))
        coord_max[2] = max(coord_max[2], float(volumes[5].max()))

    if not np.isfinite([seismic_min, seismic_max]).all():
        raise RuntimeError("Failed to scan global seismic min/max values.")
    if not np.isfinite(coord_min).all() or not np.isfinite(coord_max).all():
        raise RuntimeError("Failed to scan global coordinate min/max values.")
    return seismic_min, seismic_max, coord_min, coord_max


def normalize_volume_preserve_zero(volume, minimum, maximum):
    """Scale one volume by one global max-absolute value."""
    volume = volume.astype(np.float32, copy=True)
    scale = max(abs(float(minimum)), abs(float(maximum)))
    if scale > 0:
        volume /= scale
    else:
        volume.fill(0.0)
    return volume


def normalize_coordinate(values, minimum, maximum):
    """Map a coordinate to [-1, 1] using one shared global range."""
    denominator = float(maximum) - float(minimum)
    if denominator == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (
        2.0 * (values.astype(np.float32) - float(minimum)) / denominator - 1.0
    ).astype(np.float32)


def preprocess_volume(
    volumes,
    seismic_min,
    seismic_max,
    coord_min,
    coord_max,
    normalize,
):
    processed = volumes.astype(np.float32, copy=True)
    if normalize:
        processed[0] = normalize_volume_preserve_zero(
            processed[0],
            seismic_min,
            seismic_max,
        )

    # SX/RX share X statistics, SY/RY share Y statistics, and T has its own.
    processed[1] = normalize_coordinate(processed[1], coord_min[0], coord_max[0])
    processed[3] = normalize_coordinate(processed[3], coord_min[0], coord_max[0])
    processed[2] = normalize_coordinate(processed[2], coord_min[1], coord_max[1])
    processed[4] = normalize_coordinate(processed[4], coord_min[1], coord_max[1])
    processed[5] = normalize_coordinate(processed[5], coord_min[2], coord_max[2])
    return processed


def padded_shape(height, width, patch_size, overlap):
    stride = patch_size - overlap
    padded_height = patch_size + max(0, math.ceil((height - patch_size) / stride)) * stride
    padded_width = patch_size + max(0, math.ceil((width - patch_size) / stride)) * stride
    return padded_height, padded_width


def pad_six_volumes(volumes, patch_size, overlap):
    """Pad with the last trace/sample so every extracted patch is full-sized."""
    _, height, width = volumes.shape
    padded_height, padded_width = padded_shape(
        height,
        width,
        patch_size,
        overlap,
    )
    pad_bottom = padded_height - height
    pad_right = padded_width - width

    if pad_bottom == 0 and pad_right == 0:
        return volumes, (padded_height, padded_width)

    padded = np.pad(
        volumes,
        ((0, 0), (0, pad_bottom), (0, pad_right)),
        mode="edge",
    )
    return padded.astype(np.float32, copy=False), (padded_height, padded_width)


def extract_patches(volumes, patch_size, overlap, patch_processor):
    """Extract patches with the project's shared NumpyPatchProcessor."""
    padded, padded_shape_value = pad_six_volumes(
        volumes,
        patch_size,
        overlap,
    )

    patch_size_2d = (patch_size, patch_size)
    overlap_2d = (overlap, overlap)
    channel_patches = []
    positions = None

    for channel in range(padded.shape[0]):
        patches, channel_positions, channel_shape = (
            patch_processor.extract_overlapping_patches_2d(
                padded[channel],
                patch_size=patch_size_2d,
                overlap=overlap_2d,
            )
        )
        if channel_shape != padded_shape_value:
            raise RuntimeError("Patch processor returned an unexpected padded shape.")
        if positions is None:
            positions = channel_positions
        elif not np.array_equal(positions, channel_positions):
            raise RuntimeError("Six-channel patch grids are not aligned.")
        channel_patches.append(patches)

    return (
        np.stack(channel_patches, axis=1).astype(np.float32, copy=False),
        positions,
        padded_shape_value,
    )


def save_metadata(
    output_file,
    positions,
    original_shape,
    scale,
    coord_min,
    coord_max,
):
    np.savez(
        output_file,
        positions=np.asarray(positions, dtype=np.int64),
        original_shape=np.asarray(original_shape, dtype=np.int64),
        global_scale=np.asarray(scale, dtype=np.float32),
        global_coord_min=np.asarray(coord_min, dtype=np.float32),
        global_coord_max=np.asarray(coord_max, dtype=np.float32),
    )


def plot_shot_presence(present_shots, missing_shots, output_file):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join("/tmp", "seisflow_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    present = np.unique(np.asarray(present_shots, dtype=np.int64))
    missing = np.unique(np.asarray(missing_shots, dtype=np.int64))
    all_shots = np.unique(np.concatenate([present, missing]))
    if all_shots.size == 0:
        return

    first = int(all_shots.min())
    last = int(all_shots.max())
    x = np.arange(first, last + 1)
    present_x = np.intersect1d(x, present, assume_unique=True)
    missing_x = np.intersect1d(x, missing, assume_unique=True)

    fig, axis = plt.subplots(figsize=(14, 4.5))
    if present_x.size:
        axis.scatter(
            present_x,
            np.ones_like(present_x),
            s=18,
            label=f"generated ({present_x.size})",
        )
    if missing_x.size:
        axis.scatter(
            missing_x,
            np.zeros_like(missing_x),
            s=18,
            marker="x",
            label=f"not generated ({missing_x.size})",
        )
    axis.set_xlabel("shot number")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["not generated", "generated"])
    axis.set_title("Generated Shot Distribution")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def build_dataset(args):
    from core.dataset import SegyDataset
    from core.patching import NumpyPatchProcessor

    validate_args(args)
    output_dirs = build_output_dirs(args.output_dir)
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)

    dataset = SegyDataset(args.segy)
    patch_processor = NumpyPatchProcessor()
    train_indices, valid_indices = build_split_indices(
        len(dataset),
        args.valid,
        args.valid_mode,
        args.seed,
    )
    print(
        f"Shots: {len(dataset)}, train: {len(train_indices)}, "
        f"valid: {len(valid_indices)}"
    )

    seismic_min, seismic_max, coord_min, coord_max = scan_global_min_max(
        dataset,
        args,
    )
    print("Global seismic min/max after clip/slice:")
    print(f"  seismic: min={seismic_min:.7g}, max={seismic_max:.7g}")
    print("Global coordinate min/max [x, y, t]:")
    print(f"  min={coord_min}, max={coord_max}")

    seismic_scale = max(abs(seismic_min), abs(seismic_max))

    train_shot_numbers = []
    valid_shot_numbers = []

    for shot_index in range(len(dataset)):
        raw_volumes = load_six_volumes(
            dataset,
            shot_index,
            args.slice,
            args.clip,
        )
        processed = preprocess_volume(
            raw_volumes,
            seismic_min,
            seismic_max,
            coord_min,
            coord_max,
            args.normalize,
        )
        six_patches, positions, _ = extract_patches(
            processed,
            args.patch_size,
            args.overlap_size,
            patch_processor,
        )

        if len(six_patches) == 0:
            print(f"Skipped shot {shot_index:04d}: no patches")
            continue

        if shot_index in valid_indices:
            split_name = "valid"
        else:
            split_name = "train"

        seismic_patches = six_patches[:, 0]
        dim_patches = six_patches[:, 1:]
        np.save(
            os.path.join(output_dirs[split_name], f"patches_{shot_index:04d}.npy"),
            seismic_patches,
        )
        np.save(
            os.path.join(
                output_dirs[f"{split_name}_dim"],
                f"patches_{shot_index:04d}.npy",
            ),
            dim_patches,
        )
        save_metadata(
            os.path.join(
                args.output_dir,
                f"{split_name}_aux",
                f"patches_{shot_index:04d}.npz",
            ),
            positions,
            original_shape=processed.shape[1:],
            scale=seismic_scale,
            coord_min=coord_min,
            coord_max=coord_max,
        )

        shot_number = dataset.shot_keys[shot_index]
        if split_name == "valid":
            valid_shot_numbers.append(shot_number)
        else:
            train_shot_numbers.append(shot_number)

        print(
            f"Saved {split_name} shot {shot_index:04d}: "
            f"patches={len(six_patches)}, six_channel_shape={six_patches.shape[1:]}"
        )

    plot_shot_presence(
        train_shot_numbers,
        valid_shot_numbers,
        os.path.join(args.output_dir, "shot_presence.png"),
    )


if __name__ == "__main__":
    build_dataset(build_parser().parse_args())
