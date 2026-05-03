import argparse
import importlib.util
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Keep Matplotlib font/config caches local to the project so plotting also works
# on servers where the default user cache directory is unavailable or unwritable.
CACHE_DIR = PROJECT_DIR / ".matplotlib_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log_reader():
    read_log_path = Path(__file__).resolve().parent / "read_log.py"
    spec = importlib.util.spec_from_file_location("seisflow_read_log", read_log_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LogReader


def build_parser():
    parser = argparse.ArgumentParser(
        description="Split and plot metrics from a SimpleLogger2 log file."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Explicit log file path, e.g. log.txt or legacy train.log.",
    )
    parser.add_argument(
        "--channel",
        default="T",
        help="One-character row prefix to analyze. Default: T.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root output directory. Default: <log_stem>_analysis beside the log file.",
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


def numeric_columns(rows, columns):
    result = []
    for column in columns:
        if column == "log_index":
            continue
        if any(isinstance(row.get(column), (int, float)) for row in rows):
            result.append(column)
    return result


def safe_filename(name):
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe.strip("._") or "metric"


def plot_each_numeric_column(
    rows,
    columns,
    figures_dir,
    dpi,
    *,
    x_key=None,
    skip_columns=None,
):
    if x_key is None:
        x_key = "global_step" if "global_step" in columns else "log_index"
    skip_columns = set(skip_columns or [])
    output_paths = {}

    for column in numeric_columns(rows, columns):
        if column in skip_columns or column == x_key:
            continue
        x, y = finite_xy(rows, x_key, column)
        if not x:
            continue

        fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        axis.plot(x, y, linewidth=0.8)
        axis.set_title(f"{column} by {x_key}")
        axis.set_xlabel(x_key)
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.25)

        output_path = figures_dir / f"{safe_filename(column)}.png"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        output_paths[column] = output_path

    return output_paths


def epoch_mean_rows(rows, columns):
    if "epoch" not in columns:
        return []

    metric_columns = [
        column
        for column in numeric_columns(rows, columns)
        if column not in {"epoch", "log_index"}
    ]
    grouped = {}
    for row in rows:
        epoch = row.get("epoch")
        if not isinstance(epoch, int):
            continue
        grouped.setdefault(epoch, {column: [] for column in metric_columns})
        for column in metric_columns:
            value = row.get(column)
            if isinstance(value, (int, float)):
                grouped[epoch][column].append(float(value))

    epoch_rows = []
    for epoch in sorted(grouped):
        epoch_row = {"epoch": epoch}
        for column, values in grouped[epoch].items():
            if values:
                epoch_row[column] = sum(values) / len(values)
        epoch_rows.append(epoch_row)
    return epoch_rows


def plot_epoch_numeric_columns(rows, columns, figures_dir, dpi):
    epoch_rows = epoch_mean_rows(rows, columns)
    if not epoch_rows:
        return {}
    return plot_each_numeric_column(
        epoch_rows,
        columns,
        figures_dir,
        dpi,
        x_key="epoch",
        skip_columns={"epoch", "log_index"},
    )


def write_summary(rows, columns, output_dir, step_figures, epoch_figures):
    summary_path = output_dir / "summary.txt"
    last_row = rows[-1] if rows else {}
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"num_rows {len(rows)}\n")
        f.write(f"num_step_figures {len(step_figures)}\n")
        f.write(f"num_epoch_figures {len(epoch_figures)}\n")
        for column in numeric_columns(rows, columns):
            values = [
                row[column]
                for row in rows
                if isinstance(row.get(column), (int, float))
            ]
            if not values:
                continue
            f.write(f"{column}_last {last_row.get(column, '')}\n")
            f.write(f"{column}_min {min(values)}\n")
            f.write(f"{column}_max {max(values)}\n")
    return summary_path


def analyze(args):
    log_path = Path(args.path)
    if not log_path.is_file():
        raise FileNotFoundError(f"--path must point to a log file: {log_path}")

    LogReader = load_log_reader()
    reader = LogReader(log_path, channel=args.channel)

    run_dir = reader.path.parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / f"{reader.path.stem}_analysis"
    )
    columns_dir = output_dir / "columns"
    figures_dir = output_dir / "figures"
    step_figures_dir = figures_dir / "step"
    epoch_figures_dir = figures_dir / "epoch"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    step_figures_dir.mkdir(parents=True, exist_ok=True)
    epoch_figures_dir.mkdir(parents=True, exist_ok=True)

    columns_dir = reader.export_columns(columns_dir)
    rows = reader.to_rows()
    step_figure_paths = plot_each_numeric_column(
        rows,
        reader.columns,
        step_figures_dir,
        args.dpi,
    )
    epoch_figure_paths = plot_epoch_numeric_columns(
        rows,
        reader.columns,
        epoch_figures_dir,
        args.dpi,
    )
    outputs = {
        "output_dir": output_dir,
        "columns_dir": columns_dir,
        "figures_dir": figures_dir,
        "step_figures_dir": step_figures_dir,
        "epoch_figures_dir": epoch_figures_dir,
        "num_step_figures": len(step_figure_paths),
        "num_epoch_figures": len(epoch_figure_paths),
        "summary": write_summary(
            rows,
            reader.columns,
            output_dir,
            step_figure_paths,
            epoch_figure_paths,
        ),
    }
    return outputs


def main():
    args = build_parser().parse_args()
    outputs = analyze(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
