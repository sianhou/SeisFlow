#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "psnr": ("PSNR (dB)", True),
    "ssim": ("SSIM", True),
    "mse": ("MSE", False),
    "mae": ("MAE", False),
    "max_abs_error": ("Max absolute error", False),
}


def parse_metric(path):
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r"(shot_\d+):\s*([-+0-9.eE]+)", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def load_results(root, directory_pattern):
    results = {}
    for directory in sorted(root.glob(directory_pattern)):
        epoch = int(re.search(r"(\d+)$", directory.name).group(1))
        results[epoch] = {
            metric: parse_metric(directory / f"{metric}.txt")
            for metric in METRICS
        }
    return results


def epoch_means(results, metric):
    epochs = sorted(results)
    means = np.array([
        np.mean(list(results[epoch][metric].values()))
        for epoch in epochs
    ])
    return np.asarray(epochs), means


def save_figure(figure, output_dir, name):
    path = output_dir / name
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(path)


def plot_metric_curves(augmented, dit_ema, dit_raw, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {
        "AugmentedDiT EMA": "#d1495b",
        "DiT EMA": "#3066be",
        "DiT raw": "#7f8c8d",
    }
    series = {
        "AugmentedDiT EMA": augmented,
        "DiT EMA": dit_ema,
        "DiT raw": dit_raw,
    }

    for axis, (metric, (label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        for series_name, results in series.items():
            epochs, means = epoch_means(results, metric)
            axis.plot(
                epochs,
                means,
                marker="o",
                markersize=3,
                linewidth=2 if series_name != "DiT raw" else 1.4,
                linestyle="-" if series_name != "DiT raw" else "--",
                color=colors[series_name],
                alpha=1.0 if series_name != "DiT raw" else 0.75,
                label=series_name,
            )
            best_index = np.argmax(means) if higher_is_better else np.argmin(means)
            axis.scatter(
                epochs[best_index],
                means[best_index],
                marker="*",
                s=100,
                color=colors[series_name],
                zorder=4,
            )
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean over 75 shots")
        axis.grid(alpha=0.25)

    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Reconstruction metrics across checkpoints", fontsize=16)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(figure, output_dir, "metrics_vs_epoch.png")


def plot_epoch_summary(augmented, dit_ema, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    model_names = ["AugmentedDiT", "DiT"]
    colors = ["#d1495b", "#3066be"]

    for axis, (metric, (label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        summaries = []
        for results in (augmented, dit_ema):
            _, means = epoch_means(results, metric)
            best = np.max(means) if higher_is_better else np.min(means)
            worst = np.min(means) if higher_is_better else np.max(means)
            summaries.append([best, worst, np.mean(means)])

        x = np.arange(3)
        width = 0.36
        for model_index, (model_name, color) in enumerate(zip(model_names, colors)):
            values = np.asarray(summaries[model_index])
            bars = axis.bar(
                x + (model_index - 0.5) * width,
                values,
                width,
                color=color,
                label=model_name,
            )
            axis.bar_label(bars, fmt="%.4g", fontsize=8, padding=2)
        axis.set_xticks(x, ["Best", "Worst", "Epoch mean"])
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)

    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Best, worst, and average checkpoint performance", fontsize=16)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(figure, output_dir, "epoch_best_worst_average.png")


def best_psnr_epoch(results):
    return max(
        results,
        key=lambda epoch: np.mean(list(results[epoch]["psnr"].values())),
    )


def plot_shot_distributions(augmented, dit_ema, output_dir):
    aug_epoch = best_psnr_epoch(augmented)
    dit_epoch = best_psnr_epoch(dit_ema)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    rng = np.random.default_rng(0)

    for axis, (metric, (label, _)) in zip(axes.flat, METRICS.items()):
        aug_values = np.asarray(list(augmented[aug_epoch][metric].values()))
        dit_values = np.asarray(list(dit_ema[dit_epoch][metric].values()))
        box = axis.boxplot(
            [aug_values, dit_values],
            tick_labels=["AugmentedDiT", "DiT"],
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        for patch, color in zip(box["boxes"], ["#d1495b", "#3066be"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for index, (values, color) in enumerate(
            zip([aug_values, dit_values], ["#d1495b", "#3066be"]), start=1
        ):
            jitter = rng.normal(index, 0.045, size=len(values))
            axis.scatter(jitter, values, s=10, alpha=0.35, color=color)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)

    axes.flat[-1].axis("off")
    figure.suptitle(
        f"Per-shot distributions at each model's best-PSNR checkpoint "
        f"(AugmentedDiT {aug_epoch}, DiT {dit_epoch})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, output_dir, "best_checkpoint_shot_distributions.png")


def plot_paired_improvements(augmented, dit_ema, output_dir):
    aug_epoch = best_psnr_epoch(augmented)
    dit_epoch = best_psnr_epoch(dit_ema)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))

    for axis, (metric, (label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        shots = sorted(set(augmented[aug_epoch][metric]) & set(dit_ema[dit_epoch][metric]))
        aug_values = np.asarray([augmented[aug_epoch][metric][shot] for shot in shots])
        dit_values = np.asarray([dit_ema[dit_epoch][metric][shot] for shot in shots])
        improvement = aug_values - dit_values if higher_is_better else dit_values - aug_values
        colors = np.where(improvement >= 0, "#d1495b", "#3066be")
        axis.bar(np.arange(len(shots)), improvement, color=colors, width=0.9)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(label)
        axis.set_xlabel("Shot index (sorted by shot ID)")
        axis.set_ylabel("Improvement; positive favors AugmentedDiT")
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.02,
            0.96,
            f"wins: {(improvement > 0).sum()}/{len(improvement)}\n"
            f"mean: {improvement.mean():.4g}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes.flat[-1].axis("off")
    figure.suptitle("Paired per-shot improvement at epoch 2000", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output_dir, "paired_improvement_by_shot.png")


def plot_final_summary(augmented, dit_ema, output_dir):
    epoch = 2000
    labels = []
    relative_improvements = []
    win_rates = []
    tie_rates = []

    for metric, (label, higher_is_better) in METRICS.items():
        shots = sorted(set(augmented[epoch][metric]) & set(dit_ema[epoch][metric]))
        aug_values = np.asarray([augmented[epoch][metric][shot] for shot in shots])
        dit_values = np.asarray([dit_ema[epoch][metric][shot] for shot in shots])
        relative = (aug_values.mean() - dit_values.mean()) / dit_values.mean() * 100
        if not higher_is_better:
            relative = -relative
        improvement = aug_values - dit_values if higher_is_better else dit_values - aug_values
        labels.append(label)
        relative_improvements.append(relative)
        win_rates.append(np.mean(improvement > 0) * 100)
        tie_rates.append(np.mean(np.isclose(improvement, 0)) * 100)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(labels))
    colors = ["#d1495b" if value >= 0 else "#3066be" for value in relative_improvements]
    bars = axes[0].bar(x, relative_improvements, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Relative improvement (%)")
    axes[0].set_title("Mean metric improvement at epoch 2000")
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].grid(axis="y", alpha=0.25)

    bars = axes[1].bar(x, win_rates, color="#d1495b", label="AugmentedDiT wins")
    axes[1].bar(x, tie_rates, bottom=win_rates, color="#bfc5ca", label="Ties")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Shots (%)")
    axes[1].set_title("Paired win and tie rates over 75 shots")
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("AugmentedDiT versus DiT-EMA summary", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output_dir, "final_metric_improvement_and_win_rate.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented_dir", type=Path, required=True)
    parser.add_argument("--dit_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    augmented = load_results(args.augmented_dir, "diff_recon_shot_epoch_*")
    dit_ema = load_results(args.dit_dir, "diff_recon_shot_ema_epoch_*")
    dit_raw = load_results(args.dit_dir, "diff_recon_shot_no_ema_epoch_*")

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_metric_curves(augmented, dit_ema, dit_raw, args.output_dir)
    plot_epoch_summary(augmented, dit_ema, args.output_dir)
    plot_shot_distributions(augmented, dit_ema, args.output_dir)
    plot_paired_improvements(augmented, dit_ema, args.output_dir)
    plot_final_summary(augmented, dit_ema, args.output_dir)


if __name__ == "__main__":
    main()
