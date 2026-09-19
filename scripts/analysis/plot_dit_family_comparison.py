#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_augmented_dit_vs_dit import METRICS, epoch_means, load_results


COLORS = {
    "AugmentedDiT": "#d1495b",
    "DiT-P4": "#3066be",
    "PixelDiT-P4": "#2a9d8f",
    "PixelDiT-P16": "#e9c46a",
    "DiT-P4 raw": "#7f8c8d",
}


def metric_values(results, epoch, metric):
    shots = sorted(results[epoch][metric])
    return shots, np.asarray([results[epoch][metric][shot] for shot in shots])


def best_epoch(results, metric):
    epochs, means = epoch_means(results, metric)
    higher_is_better = METRICS[metric][1]
    index = int(np.argmax(means) if higher_is_better else np.argmin(means))
    return int(epochs[index]), float(means[index])


def best_psnr_epoch(results):
    return best_epoch(results, "psnr")[0]


def save_figure(figure, output_dir, name):
    path = output_dir / name
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(path)


def plot_metric_curves(primary, dit_raw, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    series = dict(primary)
    series["DiT-P4 raw"] = dit_raw

    for axis, (metric, (label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        for name, results in series.items():
            epochs, means = epoch_means(results, metric)
            is_raw = name.endswith("raw")
            axis.plot(
                epochs,
                means,
                marker=None if is_raw else "o",
                markersize=3,
                linewidth=1.3 if is_raw else 2,
                linestyle="--" if is_raw else "-",
                color=COLORS[name],
                alpha=0.7 if is_raw else 1.0,
                label=name,
            )
            index = int(np.argmax(means) if higher_is_better else np.argmin(means))
            if not is_raw:
                axis.scatter(
                    epochs[index], means[index], marker="*", s=90,
                    color=COLORS[name], zorder=4,
                )
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean over 75 shots")
        axis.grid(alpha=0.25)

    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("DiT-family reconstruction metrics across checkpoints", fontsize=16)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(figure, output_dir, "metrics_vs_epoch_all_models.png")


def plot_best_checkpoint_distributions(primary, output_dir):
    names = list(primary)
    epochs = {name: best_psnr_epoch(results) for name, results in primary.items()}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    rng = np.random.default_rng(0)

    for axis, (metric, (label, _)) in zip(axes.flat, METRICS.items()):
        values = [
            metric_values(primary[name], epochs[name], metric)[1]
            for name in names
        ]
        box = axis.boxplot(
            values,
            tick_labels=names,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D", "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )
        for patch, name in zip(box["boxes"], names):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.75)
        for index, (model_values, name) in enumerate(zip(values, names), start=1):
            jitter = rng.normal(index, 0.035, size=len(model_values))
            axis.scatter(
                jitter, model_values, s=8, alpha=0.22, color=COLORS[name]
            )
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)

    axes.flat[-1].axis("off")
    epoch_text = ", ".join(f"{name}: {epoch}" for name, epoch in epochs.items())
    figure.suptitle(
        f"Per-shot metrics at each model's best-PSNR checkpoint\n{epoch_text}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir, "best_psnr_checkpoint_distributions.png")


def plot_psnr_deltas(primary, output_dir):
    baseline_name = "DiT-P4"
    baseline_epoch = best_psnr_epoch(primary[baseline_name])
    shots, baseline = metric_values(primary[baseline_name], baseline_epoch, "psnr")
    comparison_names = [name for name in primary if name != baseline_name]
    figure, axes = plt.subplots(len(comparison_names), 1, figsize=(17, 9), sharex=True)

    for axis, name in zip(axes, comparison_names):
        epoch = best_psnr_epoch(primary[name])
        values = np.asarray([primary[name][epoch]["psnr"][shot] for shot in shots])
        delta = values - baseline
        colors = np.where(delta >= 0, COLORS[name], COLORS[baseline_name])
        axis.bar(np.arange(len(shots)), delta, width=0.9, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("ΔPSNR (dB)")
        axis.set_title(
            f"{name} epoch {epoch} vs {baseline_name} epoch {baseline_epoch}: "
            f"mean {delta.mean():+.3f} dB, wins {(delta > 0).sum()}/75"
        )
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Shot index sorted by shot ID")
    figure.suptitle("Paired per-shot PSNR differences at best checkpoints", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output_dir, "psnr_delta_vs_dit_by_shot.png")


def plot_psnr_summary(primary, output_dir):
    names = list(primary)
    epochs = [best_psnr_epoch(primary[name]) for name in names]
    values = [metric_values(primary[name], epoch, "psnr")[1] for name, epoch in zip(names, epochs)]
    means = np.asarray([value.mean() for value in values])
    minima = np.asarray([value.min() for value in values])
    maxima = np.asarray([value.max() for value in values])
    lower = means - minima
    upper = maxima - means

    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    bars = axis.bar(x, means, color=[COLORS[name] for name in names])
    axis.errorbar(
        x, means, yerr=np.vstack((lower, upper)), fmt="none",
        ecolor="black", capsize=5, linewidth=1.3,
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3)
    axis.set_xticks(x, [f"{name}\nepoch {epoch}" for name, epoch in zip(names, epochs)])
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Best mean PSNR with per-shot minimum and maximum")
    axis.set_ylim(max(0, minima.min() - 2), maxima.max() + 2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_dir, "best_psnr_ranking.png")


def write_epoch_means(primary, dit_raw, output_dir):
    path = output_dir / "epoch_metric_means.csv"
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("model", "epoch", *METRICS))
        for name, results in list(primary.items()) + [("DiT-P4 raw", dit_raw)]:
            for epoch in sorted(results):
                writer.writerow(
                    (
                        name,
                        epoch,
                        *[
                            np.mean(list(results[epoch][metric].values()))
                            for metric in METRICS
                        ],
                    )
                )
    print(path)


def write_detailed_statistics(primary, output_dir):
    path = output_dir / "best_checkpoint_shot_statistics.csv"
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "model", "best_psnr_epoch", "metric", "mean", "minimum",
                "maximum", "best_shot", "best_value", "worst_shot",
                "worst_value",
            )
        )
        for name, results in primary.items():
            epoch = best_psnr_epoch(results)
            for metric, (_, higher_is_better) in METRICS.items():
                shots, values = metric_values(results, epoch, metric)
                best_index = int(
                    np.argmax(values) if higher_is_better else np.argmin(values)
                )
                worst_index = int(
                    np.argmin(values) if higher_is_better else np.argmax(values)
                )
                writer.writerow(
                    (
                        name, epoch, metric, values.mean(), values.min(),
                        values.max(), shots[best_index], values[best_index],
                        shots[worst_index], values[worst_index],
                    )
                )
    print(path)

    baseline_name = "DiT-P4"
    baseline_epoch = best_psnr_epoch(primary[baseline_name])
    pair_path = output_dir / "best_checkpoint_pairwise_vs_dit.csv"
    with pair_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "model", "model_epoch", "metric", "model_mean", "dit_mean",
                "raw_delta", "favorable_delta", "wins", "ties", "losses",
            )
        )
        for name, results in primary.items():
            epoch = best_psnr_epoch(results)
            for metric, (_, higher_is_better) in METRICS.items():
                shots, baseline = metric_values(
                    primary[baseline_name], baseline_epoch, metric
                )
                values = np.asarray(
                    [results[epoch][metric][shot] for shot in shots]
                )
                raw_delta = values.mean() - baseline.mean()
                favorable = raw_delta if higher_is_better else -raw_delta
                paired = values - baseline if higher_is_better else baseline - values
                writer.writerow(
                    (
                        name, epoch, metric, values.mean(), baseline.mean(),
                        raw_delta, favorable, int(np.sum(paired > 0)),
                        int(np.sum(np.isclose(paired, 0))), int(np.sum(paired < 0)),
                    )
                )
    print(pair_path)


def write_summary(primary, dit_raw, output_dir):
    baseline_name = "DiT-P4"
    baseline_epoch = best_psnr_epoch(primary[baseline_name])
    baseline_shots, baseline_psnr = metric_values(
        primary[baseline_name], baseline_epoch, "psnr"
    )
    lines = [
        "# DiT系列 i64 / NeRF bands0 / EMA 重建统计",
        "",
        "四个EMA主模型均包含epoch 100–2000的20个checkpoint，每个checkpoint统计同一75炮。",
        "",
        "## Epoch 2000平均指标",
        "",
        "| 模型 | PSNR (dB) | SSIM | MSE | MAE | 最大绝对误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, results in primary.items():
        means = {
            metric: np.mean(list(results[2000][metric].values()))
            for metric in METRICS
        }
        lines.append(
            f"| {name} | {means['psnr']:.4f} | {means['ssim']:.5f} | "
            f"{means['mse']:.5f} | {means['mae']:.5f} | "
            f"{means['max_abs_error']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## 各模型最佳PSNR checkpoint",
            "",
            "| 模型 | Epoch | 平均PSNR | 最差炮 | 最差PSNR | 最好炮 | 最好PSNR | 相对DiT平均差 | 胜出炮数 |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|",
        ]
    )
    for name, results in primary.items():
        epoch = best_psnr_epoch(results)
        shots, values = metric_values(results, epoch, "psnr")
        aligned = np.asarray([results[epoch]["psnr"][shot] for shot in baseline_shots])
        delta = aligned - baseline_psnr
        relative_text = "—" if name == baseline_name else f"{delta.mean():+.4f} dB"
        wins_text = "—" if name == baseline_name else f"{(delta > 0).sum()}/75"
        lines.append(
            f"| {name} | {epoch} | {values.mean():.4f} | "
            f"{shots[int(np.argmin(values))]} | {values.min():.4f} | "
            f"{shots[int(np.argmax(values))]} | {values.max():.4f} | "
            f"{relative_text} | {wins_text} |"
        )

    dit_ema_final = {
        metric: np.mean(list(primary[baseline_name][2000][metric].values()))
        for metric in METRICS
    }
    dit_raw_final = {
        metric: np.mean(list(dit_raw[2000][metric].values()))
        for metric in METRICS
    }
    lines.extend(
        [
            "",
            "## DiT EMA参考",
            "",
            "| 权重 | PSNR (dB) | SSIM | MSE | MAE | 最大绝对误差 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| EMA | {dit_ema_final['psnr']:.4f} | {dit_ema_final['ssim']:.5f} | {dit_ema_final['mse']:.5f} | {dit_ema_final['mae']:.5f} | {dit_ema_final['max_abs_error']:.5f} |",
            f"| 非EMA | {dit_raw_final['psnr']:.4f} | {dit_raw_final['ssim']:.5f} | {dit_raw_final['mse']:.5f} | {dit_raw_final['mae']:.5f} | {dit_raw_final['max_abs_error']:.5f} |",
            "",
            "## 结论",
            "",
            "- AugmentedDiT最终和最佳平均PSNR最高，仍维持相对传统DiT约0.22 dB的轻微优势。",
            "- PixelDiT-P4在训练前中期收敛最快，epoch 900已达到22.018 dB，但后期平台化；最佳PSNR为22.594 dB，低于DiT 0.168 dB，仅在75炮中的18炮胜出。",
            "- PixelDiT-P4的最终SSIM最高（0.75733），说明局部结构相似性较好，但MSE和PSNR没有同步改善。",
            "- PixelDiT-P16最佳PSNR只有19.343 dB，比DiT低3.419 dB，75炮全部落后；64×64输入配16×16 patch不适合作为当前主配置。",
            "- PixelDiT-P4证明pixel-level分支可以显著加快早期学习，但当前配置下没有转化为更高的最终整体精度。",
            "",
        ]
    )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines))
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented_dir", type=Path, required=True)
    parser.add_argument("--dit_dir", type=Path, required=True)
    parser.add_argument("--pixeldit_p4_dir", type=Path, required=True)
    parser.add_argument("--pixeldit_p16_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary = {
        "AugmentedDiT": load_results(
            args.augmented_dir, "diff_recon_shot_epoch_*"
        ),
        "DiT-P4": load_results(args.dit_dir, "diff_recon_shot_ema_epoch_*"),
        "PixelDiT-P4": load_results(
            args.pixeldit_p4_dir, "diff_recon_shot_ema_epoch_*"
        ),
        "PixelDiT-P16": load_results(
            args.pixeldit_p16_dir, "diff_recon_shot_ema_epoch_*"
        ),
    }
    dit_raw = load_results(args.dit_dir, "diff_recon_shot_no_ema_epoch_*")

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_metric_curves(primary, dit_raw, args.output_dir)
    plot_best_checkpoint_distributions(primary, args.output_dir)
    plot_psnr_deltas(primary, args.output_dir)
    plot_psnr_summary(primary, args.output_dir)
    write_epoch_means(primary, dit_raw, args.output_dir)
    write_detailed_statistics(primary, args.output_dir)
    write_summary(primary, dit_raw, args.output_dir)


if __name__ == "__main__":
    main()
