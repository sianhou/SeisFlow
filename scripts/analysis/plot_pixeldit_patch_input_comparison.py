#!/usr/bin/env python3
"""Compare PixelDiT patch sizes and input sizes from DiffShot metric files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_augmented_dit_vs_dit import METRICS, epoch_means, load_results


COLORS = {
    "P2-i64": "#d1495b",
    "P4-i64": "#3066be",
    "P16-i64": "#e9c46a",
    "P4-i128": "#2a9d8f",
}


def best_psnr_epoch(results):
    epochs, means = epoch_means(results, "psnr")
    return int(epochs[int(np.argmax(means))])


def aligned_values(results, epoch, metric, shots):
    return np.asarray([results[epoch][metric][shot] for shot in shots])


def save_figure(figure, output_dir, filename):
    figure.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_curves(models, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (metric, (label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        for name, results in models.items():
            epochs, means = epoch_means(results, metric)
            axis.plot(epochs, means, marker="o", ms=3, lw=1.8, color=COLORS[name], label=name)
            index = int(np.argmax(means) if higher_is_better else np.argmin(means))
            axis.scatter(epochs[index], means[index], marker="*", s=95, color=COLORS[name], zorder=4)
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean over 75 shots")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("PixelDiT: patch size and input size comparison", fontsize=16)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(figure, output_dir, "pixeldit_metrics_vs_epoch.png")


def plot_best_distributions(models, output_dir):
    names = list(models)
    epochs = {name: best_psnr_epoch(results) for name, results in models.items()}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (metric, (label, _)) in zip(axes.flat, METRICS.items()):
        values = [np.asarray(list(models[name][epochs[name]][metric].values())) for name in names]
        box = axis.boxplot(values, tick_labels=names, patch_artist=True, showmeans=True)
        for patch, name in zip(box["boxes"], names):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.75)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    epoch_text = ", ".join(f"{name}: {epoch}" for name, epoch in epochs.items())
    figure.suptitle(f"Per-shot metrics at best-PSNR checkpoints\n{epoch_text}", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir, "pixeldit_best_checkpoint_distributions.png")


def plot_paired_psnr(models, output_dir):
    baseline_name = "P4-i64"
    baseline_epoch = best_psnr_epoch(models[baseline_name])
    shots = sorted(models[baseline_name][baseline_epoch]["psnr"])
    baseline = aligned_values(models[baseline_name], baseline_epoch, "psnr", shots)
    comparisons = [name for name in models if name != baseline_name]
    figure, axes = plt.subplots(len(comparisons), 1, figsize=(16, 8.5), sharex=True)
    for axis, name in zip(axes, comparisons):
        epoch = best_psnr_epoch(models[name])
        values = aligned_values(models[name], epoch, "psnr", shots)
        delta = values - baseline
        axis.bar(np.arange(len(shots)), delta, color=np.where(delta >= 0, COLORS[name], COLORS[baseline_name]))
        axis.axhline(0, color="black", lw=0.8)
        axis.set_ylabel("delta PSNR")
        axis.set_title(f"{name} vs P4-i64: mean {delta.mean():+.3f} dB, wins {(delta > 0).sum()}/75")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Shot index sorted by shot ID")
    figure.tight_layout()
    save_figure(figure, output_dir, "pixeldit_paired_psnr_vs_p4_i64.png")


def plot_hard_shots(models, output_dir):
    epochs = {name: best_psnr_epoch(results) for name, results in models.items()}
    shots = sorted(models["P4-i128"][epochs["P4-i128"]]["psnr"])
    ranking = sorted(
        shots,
        key=lambda shot: models["P4-i128"][epochs["P4-i128"]]["psnr"][shot],
    )[:10]
    x = np.arange(len(ranking))
    width = 0.2
    figure, axis = plt.subplots(figsize=(15, 5.5))
    for index, name in enumerate(models):
        values = aligned_values(models[name], epochs[name], "psnr", ranking)
        axis.bar(x + (index - 1.5) * width, values, width, label=name, color=COLORS[name])
    axis.set_xticks(x, ranking, rotation=35)
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Ten hardest shots under P4-i128")
    axis.legend(ncol=4)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_dir, "pixeldit_hard_shots.png")


def write_outputs(models, output_dir):
    epochs = {name: best_psnr_epoch(results) for name, results in models.items()}
    baseline_name = "P4-i64"
    baseline_epoch = epochs[baseline_name]
    shots = sorted(models[baseline_name][baseline_epoch]["psnr"])

    with (output_dir / "best_checkpoint_statistics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "epoch", "metric", "mean", "minimum", "maximum", "worst_shot", "best_shot"))
        for name, results in models.items():
            epoch = epochs[name]
            for metric, (_, higher_is_better) in METRICS.items():
                metric_shots = sorted(results[epoch][metric])
                values = aligned_values(results, epoch, metric, metric_shots)
                worst = int(np.argmin(values) if higher_is_better else np.argmax(values))
                best = int(np.argmax(values) if higher_is_better else np.argmin(values))
                writer.writerow((name, epoch, metric, values.mean(), values.min(), values.max(), metric_shots[worst], metric_shots[best]))

    pair_rows = []
    for name, results in models.items():
        epoch = epochs[name]
        for metric, (_, higher_is_better) in METRICS.items():
            values = aligned_values(results, epoch, metric, shots)
            baseline = aligned_values(models[baseline_name], baseline_epoch, metric, shots)
            favorable = values - baseline if higher_is_better else baseline - values
            pair_rows.append((name, epoch, metric, values.mean(), baseline.mean(), values.mean() - baseline.mean(),
                              int(np.sum(favorable > 0)), int(np.sum(np.isclose(favorable, 0))), int(np.sum(favorable < 0))))
    with (output_dir / "pairwise_vs_p4_i64.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "epoch", "metric", "model_mean", "p4_i64_mean", "raw_delta", "wins", "ties", "losses"))
        writer.writerows(pair_rows)

    means = {}
    for name, results in models.items():
        epoch = epochs[name]
        means[name] = {metric: float(np.mean(list(results[epoch][metric].values()))) for metric in METRICS}
    lines = [
        "# PixelDiT patch/input comparison", "",
        "All results use EMA and the same 75 reconstructed shots. Each model is reported at its best mean-PSNR checkpoint.", "",
        "| Model | Best epoch | PSNR | SSIM | MSE | MAE | Max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in models:
        value = means[name]
        lines.append(f"| {name} | {epochs[name]} | {value['psnr']:.4f} | {value['ssim']:.5f} | {value['mse']:.5f} | {value['mae']:.5f} | {value['max_abs_error']:.5f} |")
    lines.extend(["", "## Main comparisons", ""])
    for name in ("P2-i64", "P4-i128"):
        value, base = means[name], means[baseline_name]
        lines.append(
            f"- {name} vs P4-i64: PSNR {value['psnr'] - base['psnr']:+.4f} dB; "
            f"SSIM {value['ssim'] - base['ssim']:+.5f}; MSE reduced by "
            f"{(base['mse'] - value['mse']) / base['mse'] * 100:.2f}%; MAE reduced by "
            f"{(base['mae'] - value['mae']) / base['mae'] * 100:.2f}%."
        )
    value, base = means["P4-i128"], means["P2-i64"]
    lines.append(
        f"- P4-i128 vs P2-i64: PSNR {value['psnr'] - base['psnr']:+.4f} dB; "
        f"MSE reduced by {(base['mse'] - value['mse']) / base['mse'] * 100:.2f}%."
    )
    lines.extend(["", "P2-i64 is missing the epoch-400 reconstruction; this does not affect its epoch-2000 optimum.", ""])
    (output_dir / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_i64", type=Path, required=True)
    parser.add_argument("--p4_i64", type=Path, required=True)
    parser.add_argument("--p16_i64", type=Path, required=True)
    parser.add_argument("--p4_i128", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "P2-i64": load_results(args.p2_i64, "diff_recon_shot_ema_epoch_*"),
        "P4-i64": load_results(args.p4_i64, "diff_recon_shot_ema_epoch_*"),
        "P16-i64": load_results(args.p16_i64, "diff_recon_shot_ema_epoch_*"),
        "P4-i128": load_results(args.p4_i128, "diff_recon_shot_ema_epoch_*"),
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_curves(models, args.output_dir)
    plot_best_distributions(models, args.output_dir)
    plot_paired_psnr(models, args.output_dir)
    plot_hard_shots(models, args.output_dir)
    write_outputs(models, args.output_dir)


if __name__ == "__main__":
    main()
