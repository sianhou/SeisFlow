"""Plot original/reconstructed shots and their difference, matched by shot number."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Match NPY files by the number after the final underscore, then "
            "plot input1, input2 and input1-input2 for each matched shot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input1_dir", required=True,
        help="Directory containing the first shot dataset.",
    )
    parser.add_argument(
        "--input2_dir", required=True,
        help="Directory containing the second shot dataset; it determines the shot IDs.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for the generated PNG plots.",
    )
    parser.add_argument(
        "--resize", nargs=2, type=int, default=None, metavar=("HEIGHT", "WIDTH"),
        help="Resize both input arrays to HEIGHT WIDTH before plotting.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of plotting threads. Defaults to a reasonable CPU-based value.",
    )
    return parser


def _shot_number(path):
    """Return the final underscore-delimited filename component without suffix."""
    return Path(path).stem.rsplit("_", 1)[-1]


def _files_by_shot_number(directory):
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")

    files = sorted(path.glob("*.npy"))
    result = {}
    for file_path in files:
        shot_number = _shot_number(file_path)
        if shot_number in result:
            raise ValueError(
                f"Duplicate shot number {shot_number!r} in {path}: "
                f"{result[shot_number]} and {file_path}"
            )
        result[shot_number] = file_path
    return result


def match_files(input1_dir, input2_dir):
    """Match input1 files to every input2 file by its final underscore number."""
    input1_files = _files_by_shot_number(input1_dir)
    input2_files = _files_by_shot_number(input2_dir)
    if not input2_files:
        raise ValueError(f"No .npy files found in {input2_dir}")

    missing_numbers = sorted(set(input2_files) - set(input1_files))
    if missing_numbers:
        print("Files not matched in input1 (shot numbers):")
        print(" ".join(missing_numbers))

    matches = [
        (shot_number, input1_files[shot_number], input2_files[shot_number])
        for shot_number in sorted(input2_files)
        if shot_number in input1_files
    ]
    return matches


def resize_array(data, resize):
    if resize is None:
        return data
    from PIL import Image

    height, width = resize
    if height <= 0 or width <= 0:
        raise ValueError("--resize values must be positive.")
    return np.asarray(
        Image.fromarray(data.astype(np.float32, copy=False)).resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def _load_2d(path):
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D NPY array in {path}, got shape {data.shape}.")
    return data.astype(np.float32, copy=False)


def calculate_metrics(reference, reconstruction):
    """Calculate reconstruction metrics before resize and display rotation."""
    error = reference.astype(np.float64) - reconstruction.astype(np.float64)
    mse = float(np.mean(error * error))
    data_range = float(np.max(reference) - np.min(reference))
    if mse == 0:
        psnr = float("inf")
    elif data_range <= 0:
        psnr = float("-inf")
    else:
        psnr = 10.0 * np.log10((data_range * data_range) / mse)
    return {
        "psnr": psnr,
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "ssim": calculate_ssim(reference, reconstruction),
    }


def _box_mean(data, window_size):
    radius = window_size // 2
    padded = np.pad(data, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    height, width = data.shape
    total = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return total / float(window_size * window_size)


def calculate_ssim(reference, reconstruction):
    """Calculate a windowed SSIM without requiring an extra image package."""
    reference = reference.astype(np.float64, copy=False)
    reconstruction = reconstruction.astype(np.float64, copy=False)
    data_range = float(np.max(reference) - np.min(reference))
    if data_range <= 0:
        return 1.0 if np.array_equal(reference, reconstruction) else 0.0

    minimum_size = min(reference.shape)
    window_size = min(11, minimum_size)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return 1.0 if np.array_equal(reference, reconstruction) else 0.0

    mean_reference = _box_mean(reference, window_size)
    mean_reconstruction = _box_mean(reconstruction, window_size)
    mean_reference_sq = _box_mean(reference * reference, window_size)
    mean_reconstruction_sq = _box_mean(reconstruction * reconstruction, window_size)
    mean_product = _box_mean(reference * reconstruction, window_size)
    variance_reference = np.maximum(mean_reference_sq - mean_reference ** 2, 0.0)
    variance_reconstruction = np.maximum(
        mean_reconstruction_sq - mean_reconstruction ** 2, 0.0
    )
    covariance = mean_product - mean_reference * mean_reconstruction

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mean_reference * mean_reconstruction + c1) * (2 * covariance + c2)
    denominator = (
        (mean_reference ** 2 + mean_reconstruction ** 2 + c1)
        * (variance_reference + variance_reconstruction + c2)
    )
    return float(np.mean(numerator / denominator))


def plot_one(shot_number, input1_path, input2_path, output_dir, resize):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/seisflow_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    input1 = _load_2d(input1_path)
    input2 = _load_2d(input2_path)
    if input1.shape != input2.shape:
        raise ValueError(
            f"Shape mismatch for shot {shot_number}: "
            f"input1={input1.shape}, input2={input2.shape}; use --resize."
        )

    metrics = calculate_metrics(input1, input2)
    difference = input1 - input2
    input1 = resize_array(input1, resize)
    input2 = resize_array(input2, resize)
    difference = resize_array(difference, resize)
    # Seismic display orientation: rotate clockwise by 90 degrees.
    input1 = np.rot90(input1, k=-1)
    input2 = np.rot90(input2, k=-1)
    difference = np.rot90(difference, k=-1)
    vmin = float(np.min(input1))
    vmax = float(np.max(input1))

    figure = Figure(figsize=(18, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    images = (input1, input2, difference)
    psnr = metrics["psnr"]
    psnr_text = "inf" if np.isinf(psnr) and psnr > 0 else f"{psnr:.4f} dB"
    titles = ("Original", "Reconstructed", f"Original - Reconstructed\nPSNR: {psnr_text}")
    for axis, data, title in zip(axes, images, titles):
        image = axis.imshow(data, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    output_path = Path(output_dir) / f"shot_{shot_number}.png"
    figure.savefig(output_path, dpi=180)
    return output_path, metrics


def write_metric_reports(output_dir, results):
    metric_labels = {
        "psnr": "PSNR",
        "mse": "MSE",
        "mae": "MAE",
        "max_abs_error": "MaxAbsError",
        "ssim": "SSIM",
    }
    for metric_name, metric_label in metric_labels.items():
        values = [(shot_number, metrics[metric_name]) for shot_number, metrics in results]
        valid_values = [value for _, value in values if not np.isnan(value)]
        report_path = Path(output_dir) / f"{metric_name}.txt"
        with report_path.open("w", encoding="utf-8") as report:
            report.write(f"{metric_label}\n")
            report.write("=" * len(metric_label) + "\n")
            for shot_number, value in values:
                report.write(f"shot_{shot_number}: {value:.10g}\n")
            if valid_values:
                report.write("\nSummary\n")
                report.write(f"max: {max(valid_values):.10g}\n")
                report.write(f"min: {min(valid_values):.10g}\n")
                report.write(f"mean: {np.mean(valid_values):.10g}\n")
            else:
                report.write("\nSummary\nmax: nan\nmin: nan\nmean: nan\n")
        print(f"Saved metric report: {report_path}")


def run(args):
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive.")
    matches = match_files(args.input1_dir, args.input2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not matches:
        print("No matched files to plot.")
        return

    workers = args.workers or min(32, (os.cpu_count() or 1) + 4)
    print(f"Matched files: {len(matches)}, workers: {workers}")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                plot_one,
                shot_number,
                input1_path,
                input2_path,
                output_dir,
                args.resize,
            ): shot_number
            for shot_number, input1_path, input2_path in matches
        }
        try:
            for index, future in enumerate(as_completed(futures), start=1):
                output_path, metrics = future.result()
                results.append((futures[future], metrics))
                psnr = metrics["psnr"]
                psnr_text = "inf" if np.isinf(psnr) and psnr > 0 else f"{psnr:.4f} dB"
                print(f"[{index}/{len(matches)}] Saved {output_path}, PSNR={psnr_text}")
        except Exception as exc:
            for future in futures:
                future.cancel()
            raise RuntimeError(
                f"Plotting failed for shot {futures[future]}"
            ) from exc
    write_metric_reports(output_dir, sorted(results))


if __name__ == "__main__":
    run(build_parser().parse_args())
