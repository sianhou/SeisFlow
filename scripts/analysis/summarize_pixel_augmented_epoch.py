#!/usr/bin/env python3
"""Summarize PixelDiT/AugmentedDiT results at one exact epoch."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METRICS = ("psnr", "ssim", "mse", "mae", "max_abs_error")
METRIC_LABELS = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "mse": "MSE",
    "mae": "MAE",
    "max_abs_error": "Max absolute error",
}


def parse_metric(path):
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r"(shot_\d+):\s*([-+0-9.eE]+)", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def display_name(root):
    return root.name.removeprefix("recon_")


def short_name(name):
    aliases = {
        "AugmentedDiTSeisDimReconNeRF_t_i64_NerfBands0": "Aug-T-P4-i64-B0 (legacy)",
        "AugmentedDiTSeisDimReconNeRF_t_p2_i64_NeRFBands0": "Aug-T-P2-i64-B0",
        "AugmentedDiTSeisDimReconNeRF_t_p4_i64_NeRFBands0": "Aug-T-P4-i64-B0",
        "PixelDiTSeisDimReconNeRF_t_p16_i64_NeRFBands0": "Pixel-T-P16-i64-B0",
        "PixelDiTSeisDimReconNeRF_t_p2_i64_NeRFBands0": "Pixel-T-P2-i64-B0",
        "PixelDiTSeisDimReconNeRF_t_p4_i128_NeRFBands0": "Pixel-T-P4-i128-B0",
        "PixelDiTSeisDimReconNeRF_t_p4_i64_NeRFBands0": "Pixel-T-P4-i64-B0",
        "PixelDiTSeisDimReconNeRF_t_p8_i256_NeRFBands0": "Pixel-T-P8-i256-B0",
    }
    return aliases.get(name, name)


def find_epoch_dir(root, epoch):
    candidates = sorted(root.glob(f"diff_recon_shot*epoch_{epoch:05d}"))
    ema = [path for path in candidates if "ema" in path.name and "no_ema" not in path.name]
    return ema[-1] if ema else (candidates[-1] if candidates else None)


def load_experiments(input_root, epoch):
    roots = sorted(
        path for path in input_root.glob("recon_*")
        if path.is_dir() and ("PixelDiT" in path.name or "AugmentedDiT" in path.name)
    )
    experiments = []
    for root in roots:
        epoch_dir = find_epoch_dir(root, epoch)
        record = {"root": root, "name": display_name(root), "epoch_dir": epoch_dir, "metrics": None}
        if epoch_dir is not None and all((epoch_dir / f"{metric}.txt").is_file() for metric in METRICS):
            metrics = {metric: parse_metric(epoch_dir / f"{metric}.txt") for metric in METRICS}
            shot_sets = [set(values) for values in metrics.values()]
            common = set.intersection(*shot_sets)
            if common:
                record["metrics"] = {
                    metric: {shot: metrics[metric][shot] for shot in sorted(common)}
                    for metric in METRICS
                }
        experiments.append(record)
    return experiments


def summarize(experiments):
    for experiment in experiments:
        metrics = experiment["metrics"]
        experiment["statistics"] = {}
        if metrics is None:
            experiment["worst_shot"] = None
            experiment["best_shot"] = None
            continue
        for metric, values in metrics.items():
            array = np.asarray(list(values.values()), dtype=np.float64)
            experiment["statistics"][metric] = {
                "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())
            }
        psnr = metrics["psnr"]
        experiment["worst_shot"] = min(psnr, key=psnr.get)
        experiment["best_shot"] = max(psnr, key=psnr.get)


def write_statistics(experiments, epoch, output_dir):
    fields = ["experiment", "epoch", "available", "num_shots"]
    fields += [f"{metric}_{stat}" for metric in METRICS for stat in ("min", "max", "mean")]
    path = output_dir / f"epoch_{epoch:05d}_statistics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in experiments:
            available = experiment["metrics"] is not None
            row = {
                "experiment": experiment["name"], "epoch": epoch if available else "",
                "available": available, "num_shots": len(experiment["metrics"]["psnr"]) if available else "",
            }
            if available:
                for metric in METRICS:
                    for stat in ("min", "max", "mean"):
                        row[f"{metric}_{stat}"] = experiment["statistics"][metric][stat]
            writer.writerow(row)
    return path


def write_selected_shots(experiments, epoch, output_dir):
    path = output_dir / f"epoch_{epoch:05d}_best_worst_psnr_shots.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "epoch", "worst_shot", "worst_psnr", "best_shot", "best_psnr"))
        for experiment in experiments:
            if experiment["metrics"] is None:
                writer.writerow((experiment["name"], "", "", "", "", ""))
                continue
            psnr = experiment["metrics"]["psnr"]
            worst, best = experiment["worst_shot"], experiment["best_shot"]
            writer.writerow((experiment["name"], epoch, worst, psnr[worst], best, psnr[best]))
    return path


def plot_metric_ranges(experiments, epoch, output_dir):
    available = [experiment for experiment in experiments if experiment["metrics"] is not None]
    names = [short_name(experiment["name"]) for experiment in available]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(available)))
    figure, axes = plt.subplots(2, 3, figsize=(20, 11))
    x = np.arange(len(available))
    for axis, metric in zip(axes.flat, METRICS):
        means = np.asarray([experiment["statistics"][metric]["mean"] for experiment in available])
        minima = np.asarray([experiment["statistics"][metric]["min"] for experiment in available])
        maxima = np.asarray([experiment["statistics"][metric]["max"] for experiment in available])
        axis.bar(x, means, color=colors, alpha=0.82)
        axis.errorbar(x, means, yerr=np.vstack((means - minima, maxima - means)), fmt="none", ecolor="black", capsize=3)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x, names, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    figure.suptitle(f"PixelDiT and AugmentedDiT statistics at epoch {epoch} (mean, min-max)", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / f"epoch_{epoch:05d}_metric_ranges.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_shot_collage(experiments, epoch, output_dir, which):
    available = [experiment for experiment in experiments if experiment["metrics"] is not None]
    figure, axes = plt.subplots(len(available), 3, figsize=(18, 3.1 * len(available)), squeeze=False)
    for row, experiment in enumerate(available):
        shot = experiment[f"{which}_shot"]
        png = experiment["epoch_dir"] / f"{shot}.png"
        image = np.asarray(Image.open(png).convert("RGB"))
        width = image.shape[1]
        bounds = (0, width // 3, 2 * width // 3, width)
        psnr = experiment["metrics"]["psnr"][shot]
        for column in range(3):
            axes[row, column].imshow(image[:, bounds[column]:bounds[column + 1]])
            axes[row, column].axis("off")
        axes[row, 0].text(
            -0.02, 0.5, f"{short_name(experiment['name'])}\n{shot}, PSNR={psnr:.4f} dB",
            transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=8,
        )
    figure.suptitle(f"Epoch {epoch}: {which} PSNR shot from each available experiment", fontsize=16, y=0.997)
    figure.subplots_adjust(left=0.18, right=0.995, top=0.985, bottom=0.01, hspace=0.08, wspace=0.01)
    path = output_dir / f"epoch_{epoch:05d}_{which}_psnr_shots.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def write_summary(experiments, epoch, output_dir):
    lines = [
        f"# PixelDiT / AugmentedDiT epoch {epoch} statistics", "",
        "Only the exact requested epoch is used. Missing checkpoints are intentionally left blank.", "",
    ]
    for metric in METRICS:
        lines.extend([
            f"## {METRIC_LABELS[metric]}", "",
            "| Experiment | Min | Mean | Max |", "|---|---:|---:|---:|",
        ])
        for experiment in experiments:
            if experiment["metrics"] is None:
                lines.append(f"| {experiment['name']} |  |  |  |")
            else:
                stats = experiment["statistics"][metric]
                lines.append(f"| {experiment['name']} | {stats['min']:.6f} | {stats['mean']:.6f} | {stats['max']:.6f} |")
        lines.append("")
    lines.extend(["## Best and worst PSNR shots", "", "| Experiment | Worst | PSNR | Best | PSNR |", "|---|---|---:|---|---:|"])
    for experiment in experiments:
        if experiment["metrics"] is None:
            lines.append(f"| {experiment['name']} |  |  |  |  |")
        else:
            psnr = experiment["metrics"]["psnr"]
            worst, best = experiment["worst_shot"], experiment["best_shot"]
            lines.append(f"| {experiment['name']} | {worst} | {psnr[worst]:.6f} | {best} | {psnr[best]:.6f} |")
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("temp/new"))
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(args.input_root, args.epoch)
    summarize(experiments)
    outputs = [
        write_statistics(experiments, args.epoch, args.output_dir),
        write_selected_shots(experiments, args.epoch, args.output_dir),
        plot_metric_ranges(experiments, args.epoch, args.output_dir),
        plot_shot_collage(experiments, args.epoch, args.output_dir, "worst"),
        plot_shot_collage(experiments, args.epoch, args.output_dir, "best"),
        write_summary(experiments, args.epoch, args.output_dir),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
