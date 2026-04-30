import argparse
import importlib.util
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".matplotlib_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path("20260429_154416_325944_vae_c4_n64_t1")
KEY_METRICS = ("loss", "mae", "mse", "rmse", "psnr")
RUNTIME_METRICS = ("grad_norm", "step_time_sec", "samples_per_sec")


def load_log_reader():
    logreader_path = Path(__file__).resolve().parent / "core" / "logging" / "logreader.py"
    spec = importlib.util.spec_from_file_location("seisflow_logreader", logreader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LogReader


def build_parser():
    parser = argparse.ArgumentParser(
        description="Split and plot metrics from a Seismic VAE SimpleLogger2 train.log."
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_RUN_DIR),
        help="Run directory or train.log path.",
    )
    parser.add_argument(
        "--channel",
        default="train",
        help="Log channel when --path is a run directory.",
    )
    parser.add_argument(
        "--columns_dir",
        default=None,
        help="Directory for split column text files. Default: <log_stem>_columns.",
    )
    parser.add_argument(
        "--figures_dir",
        default=None,
        help="Directory for generated figures. Default: <run_dir>/figures.",
    )
    parser.add_argument(
        "--dpi",
        default=160,
        type=int,
        help="Figure resolution.",
    )
    return parser


def finite_xy(rows, x_key, y_key):
    x_values = []
    y_values = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            x_values.append(x)
            y_values.append(y)
    return x_values, y_values


def epoch_means(rows, metric):
    sums = {}
    counts = {}
    for row in rows:
        epoch = row.get("epoch")
        value = row.get(metric)
        if not isinstance(epoch, int) or not isinstance(value, (int, float)):
            continue
        sums[epoch] = sums.get(epoch, 0.0) + float(value)
        counts[epoch] = counts.get(epoch, 0) + 1
    epochs = sorted(sums)
    means = [sums[epoch] / counts[epoch] for epoch in epochs]
    return epochs, means


def plot_step_metrics(rows, figures_dir, dpi):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for axis, metric in zip(axes, KEY_METRICS):
        x, y = finite_xy(rows, "global_step", metric)
        axis.plot(x, y, linewidth=0.6)
        axis.set_title(f"{metric} by step")
        axis.set_xlabel("global_step")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)

    axes[-1].axis("off")
    output_path = figures_dir / "key_metrics_by_step.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_epoch_metrics(rows, figures_dir, dpi):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for axis, metric in zip(axes, KEY_METRICS):
        epochs, means = epoch_means(rows, metric)
        axis.plot(epochs, means, linewidth=1.2)
        axis.set_title(f"mean {metric} by epoch")
        axis.set_xlabel("epoch")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)

    axes[-1].axis("off")
    output_path = figures_dir / "key_metrics_by_epoch.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_runtime_metrics(rows, figures_dir, dpi):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    for axis, metric in zip(axes, RUNTIME_METRICS):
        x, y = finite_xy(rows, "global_step", metric)
        axis.plot(x, y, linewidth=0.6)
        axis.set_title(f"{metric} by step")
        axis.set_xlabel("global_step")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)

    output_path = figures_dir / "runtime_metrics_by_step.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_summary(rows, figures_dir):
    summary_path = figures_dir / "train_log_summary.txt"
    last_row = rows[-1] if rows else {}
    best_loss = min(
        (row["loss"] for row in rows if isinstance(row.get("loss"), (int, float))),
        default=None,
    )
    best_psnr = max(
        (row["psnr"] for row in rows if isinstance(row.get("psnr"), (int, float))),
        default=None,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"num_rows {len(rows)}\n")
        f.write(f"last_epoch {last_row.get('epoch', '')}\n")
        f.write(f"last_global_step {last_row.get('global_step', '')}\n")
        f.write(f"last_loss {last_row.get('loss', '')}\n")
        f.write(f"last_mae {last_row.get('mae', '')}\n")
        f.write(f"last_mse {last_row.get('mse', '')}\n")
        f.write(f"last_rmse {last_row.get('rmse', '')}\n")
        f.write(f"last_psnr {last_row.get('psnr', '')}\n")
        f.write(f"best_step_loss {best_loss if best_loss is not None else ''}\n")
        f.write(f"best_step_psnr {best_psnr if best_psnr is not None else ''}\n")
    return summary_path


def analyze(args):
    LogReader = load_log_reader()
    reader = LogReader(args.path, channel=args.channel)

    columns_dir = reader.export_columns(args.columns_dir)
    run_dir = reader.path.parent
    figures_dir = Path(args.figures_dir) if args.figures_dir else run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = reader.to_rows()
    outputs = {
        "columns_dir": columns_dir,
        "step_metrics": plot_step_metrics(rows, figures_dir, args.dpi),
        "epoch_metrics": plot_epoch_metrics(rows, figures_dir, args.dpi),
        "runtime_metrics": plot_runtime_metrics(rows, figures_dir, args.dpi),
        "summary": write_summary(rows, figures_dir),
    }
    return outputs


def main():
    args = build_parser().parse_args()
    outputs = analyze(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
