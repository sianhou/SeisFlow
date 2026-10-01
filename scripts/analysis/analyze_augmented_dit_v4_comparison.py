"""Compare the downloaded AugmentedDiT and V4 validation shots through epoch 1000.

Run from the repository root with ``python scripts/analysis/analyze_augmented_dit_v4_comparison.py``.
The data directory contains the read-only remote snapshot and its manifest.
"""

import csv
import importlib.util
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research_logs/assets/augmented_dit_v4_comparison"
DATA = OUTPUT / "data"
EPOCHS = list(range(100, 1001, 100))
MODELS = ("baseline", "v4")
COLORS = {"baseline": "#3465a4", "v4": "#df652f"}
spec = importlib.util.spec_from_file_location(
    "snapshot_diffshot", DATA / "code_at_collection/DiffShot.py"
)
diffshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diffshot)


def write_csv(name, rows):
    """Write dictionaries in rows to the output CSV named name; return nothing."""
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(reference, reconstruction):
    """Return DiffShot and waveform metrics for two [trace, sample] arrays.

    reference is the amplitude-restored target; reconstruction is the corresponding
    model output. Extra metrics describe relative error, energy and zero-lag correlation.
    """
    values = diffshot.calculate_metrics(reference, reconstruction)
    ref = reference.astype(np.float64)
    rec = reconstruction.astype(np.float64)
    values["relative_l2"] = float(np.sqrt(np.sum((ref - rec) ** 2) / np.sum(ref ** 2)))
    values["rms_ratio"] = float(np.sqrt(np.sum(rec ** 2) / np.sum(ref ** 2)))
    values["correlation"] = float(np.corrcoef(ref.ravel(), rec.ravel())[0, 1])
    return values


def collect_metrics(shots, references, reconstructions, groups):
    """Return per-shot, per-block, overall and 50-group metric rows plus audit error.

    shots contains IDs; references maps shot IDs to arrays; reconstructions maps
    (model, epoch, shot) to arrays. groups is a list of 50 contiguous trace-index arrays.
    Overall metrics are shot means. Group metrics are means over corresponding shot
    blocks; blocks within a group have equal sizes. Also compare all five original
    per-shot metrics with the remote text files and return maximum absolute deviations.
    """
    shot_rows, block_rows, overall_rows, group_rows = [], [], [], []
    audit = {key: 0.0 for key in ("mse", "mae", "psnr", "ssim", "max_abs_error")}
    for model in MODELS:
        for epoch in EPOCHS:
            current_shots, current_blocks = [], []
            saved_metrics = {}
            for key in audit:
                text = (DATA / model / f"metrics_{epoch:05d}/{key}.txt").read_text()
                saved_metrics[key] = {
                    int(shot): float(value)
                    for shot, value in re.findall(r"shot_(\d+): ([\d.eE+-]+)", text)
                }
            for shot in shots:
                ref, rec = references[shot], reconstructions[model, epoch, shot]
                values = metrics(ref, rec)
                for key in audit:
                    audit[key] = max(audit[key], abs(values[key] - saved_metrics[key][shot]))
                current_shots.append({"model": model, "epoch": epoch, "shot": shot, **values})
                for group, indices in enumerate(groups, start=1):
                    current_blocks.append({
                        "model": model, "epoch": epoch, "shot": shot, "group": group,
                        "trace_start": int(indices[0]), "trace_stop": int(indices[-1] + 1),
                        "pixels": int(len(indices) * ref.shape[1]),
                        **metrics(ref[indices], rec[indices]),
                    })
            keys = list(values)
            overall_rows.append({"model": model, "epoch": epoch, **{
                key: float(np.mean([row[key] for row in current_shots])) for key in keys
            }})
            for group in range(1, 51):
                rows = [row for row in current_blocks if row["group"] == group]
                group_rows.append({"model": model, "epoch": epoch, "group": group, **{
                    key: float(np.mean([row[key] for row in rows])) for key in keys
                }})
            shot_rows.extend(current_shots)
            block_rows.extend(current_blocks)
    return shot_rows, block_rows, overall_rows, group_rows, audit


def save_plot(figure, name):
    """Save figure to the PNG file name under the output directory and close it."""
    figure.savefig(OUTPUT / name, dpi=150, facecolor="white")
    plt.close(figure)


def plot_curves(overall, shot_rows, group_rows, block_rows, shots):
    """Plot checkpoint curves and paired MSE maps from the four metric tables.

    overall, shot_rows, group_rows and block_rows contain aggregate, shot, group and
    shot-by-group metric dictionaries; shots specifies the heatmap row ordering.
    """
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for axis, key in zip(axes.flat, ("mse", "mae", "psnr", "ssim", "correlation", "rms_ratio")):
        for model in MODELS:
            rows = [row for row in overall if row["model"] == model]
            axis.plot(EPOCHS, [row[key] for row in rows], "o-", label=model, color=COLORS[model])
        axis.set(xlabel="Epoch", ylabel=key, title=f"Validation mean: {key}")
        axis.grid(alpha=.2)
        axis.legend()
    save_plot(figure, "overall_curves.png")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for axis, rows, field, identifiers in (
        (axes[0], shot_rows, "shot", shots),
        (axes[1], group_rows, "group", list(range(1, 51))),
    ):
        lookup = {(row["model"], row["epoch"], row[field]): row["mse"] for row in rows}
        improvement = np.array([
            [100 * (1 - lookup["v4", epoch, item] / lookup["baseline", epoch, item])
             for epoch in EPOCHS] for item in identifiers
        ])
        image = axis.imshow(improvement, cmap="RdBu", aspect="auto", vmin=-20, vmax=20)
        axis.set_xticks(range(10), EPOCHS, rotation=45)
        indices = list(range(len(identifiers))) if field == "shot" else list(range(0, 50, 5)) + [49]
        axis.set_yticks(indices, [identifiers[i] for i in indices])
        axis.set(xlabel="Epoch", ylabel=field, title="MSE reduction (%); positive = V4 better")
        figure.colorbar(image, ax=axis, extend="both")
    save_plot(figure, "paired_epoch_heatmaps.png")
    lookup = {(r["model"], r["shot"], r["group"]): r["mse"]
              for r in block_rows if r["epoch"] == 1000}
    delta = np.array([[100 * (1 - lookup["v4", shot, group] / lookup["baseline", shot, group])
                       for group in range(1, 51)] for shot in shots])
    figure, axis = plt.subplots(figsize=(16, 6), layout="constrained")
    image = axis.imshow(delta, cmap="RdBu", aspect="auto", vmin=-30, vmax=30)
    axis.set_yticks(range(len(shots)), shots)
    axis.set_xticks(np.arange(0, 50, 5), np.arange(1, 51, 5))
    axis.set(xlabel="Trace group", ylabel="Shot index", title="Epoch 1000: MSE reduction (%); positive = V4 better")
    figure.colorbar(image, ax=axis, extend="both")
    save_plot(figure, "shot_group_heatmap.png")
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), layout="constrained")
    for model in MODELS:
        rows = [r for r in group_rows if r["epoch"] == 1000 and r["model"] == model]
        axes[0].plot(range(1, 51), [r["mse"] for r in rows], "o-", label=model, color=COLORS[model])
    lookup = {(r["model"], r["group"]): r["mse"] for r in group_rows if r["epoch"] == 1000}
    delta = [100 * (1 - lookup["v4", g] / lookup["baseline", g]) for g in range(1, 51)]
    axes[1].bar(range(1, 51), delta, color=["#3465a4" if d > 0 else "#df652f" for d in delta])
    axes[0].set(ylabel="MSE", title="Epoch 1000: 50 trace groups, pooled over 15 shots")
    axes[0].legend()
    axes[1].set(xlabel="Trace group", ylabel="MSE reduction (%)")
    for axis in axes:
        axis.grid(alpha=.2)
    save_plot(figure, "group_1000.png")


def plot_reconstruction(reference, baseline, v4, title, name, boundaries=()):
    """Draw matched arrays and residuals with shared amplitude limits.

    reference, baseline and v4 are [trace, sample] arrays. title labels the comparison;
    name is the output PNG path. boundaries marks joins between different shots when
    drawing a group montage. Return nothing; no normalization or gain is applied.
    """
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    arrays = (reference, baseline, v4, reference - baseline, reference - v4, v4 - baseline)
    titles = ("Reference", "AugmentedDiT", "AugmentedDiT V4", "Reference - baseline", "Reference - V4", "V4 - baseline")
    for axis, array, label in zip(axes.flat, arrays, titles):
        image = axis.imshow(array.T, cmap="seismic", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
        axis.set(title=label, xlabel="Trace index" if not boundaries else "Concatenated trace index", ylabel="Time sample index")
        for boundary in boundaries:
            axis.axvline(boundary - .5, color="black", linewidth=.5, alpha=.5)
    figure.suptitle(title)
    figure.colorbar(image, ax=axes, label="Restored amplitude (all panels: -2 to 2; larger residuals saturated)", shrink=.75)
    save_plot(figure, name)


def plot_waveforms(shots, references, reconstructions, groups):
    """Plot all 15 shots and all 50 groups at epoch 1000, plus checkpoint montages.

    shots lists shot IDs; references and reconstructions map IDs/keys to arrays;
    groups contains each contiguous trace-index array. Returns nothing.
    """
    for subdir in ("shots", "groups", "evolution", "traces"):
        (OUTPUT / subdir).mkdir(exist_ok=True)
    for shot in shots:
        reference = references[shot]
        plot_reconstruction(reference, reconstructions["baseline", 1000, shot],
                            reconstructions["v4", 1000, shot], f"Shot {shot:04d} | EMA epoch 1000",
                            f"shots/shot_{shot:04d}.png")
        figure, axes = plt.subplots(10, 3, figsize=(12, 30), layout="constrained")
        for row, epoch in enumerate(EPOCHS):
            arrays = (reference, reconstructions["baseline", epoch, shot], reconstructions["v4", epoch, shot])
            for axis, array, label in zip(axes[row], arrays, ("Reference", "Baseline", "V4")):
                axis.imshow(array.T, cmap="seismic", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
                axis.set_title(f"{label} | epoch {epoch}", fontsize=9)
                axis.set_ylabel("Time sample")
                axis.set_xlabel("Trace index")
        figure.suptitle(f"Shot {shot:04d} | shared amplitude range [-2, 2] | no gain")
        save_plot(figure, f"evolution/shot_{shot:04d}.png")
        figure, axes = plt.subplots(3, 1, figsize=(14, 8), layout="constrained")
        for axis, trace in zip(axes, (30, 150, 270)):
            axis.plot(reference[trace], color="black", linewidth=1, label="reference")
            for model in MODELS:
                axis.plot(reconstructions[model, 1000, shot][trace], color=COLORS[model], linewidth=.9, alpha=.85, label=model)
            axis.set(title=f"Shot {shot:04d}, trace {trace} | epoch 1000", xlabel="Time sample index", ylabel="Amplitude", ylim=(-2.1, 2.1))
            axis.legend(ncol=3)
        save_plot(figure, f"traces/shot_{shot:04d}.png")
    for group, indices in enumerate(groups, start=1):
        arrays = [np.concatenate([references[s][indices] for s in shots])]
        for model in MODELS:
            arrays.append(np.concatenate([reconstructions[model, 1000, s][indices] for s in shots]))
        plot_reconstruction(*arrays,
                            f"Group {group:02d}, traces {indices[0]}-{indices[-1]} | epoch 1000 | shot order: {shots}",
                            f"groups/group_{group:02d}.png", tuple(np.arange(1, len(shots)) * len(indices)))


def write_gallery(shots, groups):
    """Write a browsable HTML index for shot IDs and trace groups, without scripts."""
    sections = ["<!doctype html><meta charset='utf-8'><title>AugmentedDiT / V4 reconstruction comparison</title>",
                "<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px}img{width:100%}a{margin-right:15px}summary{cursor:pointer;margin:15px 0}</style>",
                "<h1>AugmentedDiT 与 V4：epoch 100–1000</h1>",
                "<p>15 炮验证集；50 组为各炮的连续接收道区段，索引从 0 开始。所有重建图采用相同幅值范围 [-2,2]，无自动增益。分组图用竖线分隔不同炮，炮序见标题。残差超出色标范围时饱和显示。</p>"]
    for name in ("overall_curves", "paired_epoch_heatmaps", "shot_group_heatmap", "group_1000"):
        sections.append(f"<img loading='lazy' src='{name}.png' alt='{name}'>")
    sections.append("<h2>逐炮重建、残差、波形及 10 个 checkpoint</h2>")
    for shot in shots:
        sections.append(f"<details><summary>炮 {shot:04d}</summary><a href='evolution/shot_{shot:04d}.png'>全部 10 个 epoch</a><a href='traces/shot_{shot:04d}.png'>3 条代表道波形</a><img loading='lazy' src='shots/shot_{shot:04d}.png'></details>")
    sections.append("<h2>50 组分别比较（epoch 1000）</h2>")
    for group, indices in enumerate(groups, start=1):
        sections.append(f"<details><summary>组 {group:02d}，道 {indices[0]}–{indices[-1]}</summary><img loading='lazy' src='groups/group_{group:02d}.png'></details>")
    (OUTPUT / "index.html").write_text("\n".join(sections))


def energy_decomposition(shots, references, reconstructions):
    """Return pooled epoch-1000 error-energy terms for the listed shots.

    references maps shot IDs to targets; reconstructions maps (model, epoch, shot)
    to outputs. Terms satisfy MSE = reference_energy + prediction_energy - 2*cross_moment.
    """
    ref = np.stack([references[shot] for shot in shots]).astype(np.float64)
    result = {}
    for model in MODELS:
        rec = np.stack([reconstructions[model, 1000, shot] for shot in shots]).astype(np.float64)
        result[model] = {
            "reference_energy": float(np.mean(ref ** 2)),
            "prediction_energy": float(np.mean(rec ** 2)),
            "cross_moment": float(np.mean(ref * rec)),
            "pooled_relative_l2": float(np.sqrt(np.sum((ref - rec) ** 2) / np.sum(ref ** 2))),
            "pooled_rms_ratio": float(np.sqrt(np.sum(rec ** 2) / np.sum(ref ** 2))),
        }
    return result


def main():
    """Read the fixed snapshot and write metric CSVs, numerical summaries and plots."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    shots = manifest["shots"]
    references = {shot: np.load(DATA / f"reference/shot_{shot:04d}.npy") for shot in shots}
    reconstructions = {(model, epoch, shot): np.load(DATA / model / f"epoch_{epoch:05d}/patches_{shot:04d}.npy")
                       for model in MODELS for epoch in EPOCHS for shot in shots}
    groups = np.array_split(np.arange(301), 50)
    shot_rows, block_rows, overall, group_rows, audit = collect_metrics(shots, references, reconstructions, groups)
    for name, rows in (("shot_metrics.csv", shot_rows), ("shot_group_metrics.csv", block_rows),
                       ("overall_metrics.csv", overall), ("group_metrics.csv", group_rows)):
        write_csv(name, rows)
    write_csv("group_boundaries.csv", [{"group": i, "trace_start": int(g[0]),
                                       "trace_stop_exclusive": int(g[-1] + 1), "traces": len(g)}
                                      for i, g in enumerate(groups, start=1)])
    summary = {"remote_metric_max_abs_difference": audit, "epochs": [], "shots": [], "groups": []}
    for epoch in EPOCHS:
        pair = {r["model"]: r for r in overall if r["epoch"] == epoch}
        s = {(r["model"], r["shot"]): r for r in shot_rows if r["epoch"] == epoch}
        g = {(r["model"], r["group"]): r for r in group_rows if r["epoch"] == epoch}
        summary["epochs"].append({"epoch": epoch, "baseline": pair["baseline"], "v4": pair["v4"],
                                  "mse_reduction_pct": 100 * (1 - pair["v4"]["mse"] / pair["baseline"]["mse"]),
                                  "shot_mse_wins": sum(s["v4", x]["mse"] < s["baseline", x]["mse"] for x in shots),
                                  "group_mse_wins": sum(g["v4", x]["mse"] < g["baseline", x]["mse"] for x in range(1, 51))})
    for field, identifiers, rows in (("shot", shots, shot_rows), ("group", range(1, 51), group_rows)):
        lookup = {(r["model"], r[field]): r for r in rows if r["epoch"] == 1000}
        for item in identifiers:
            b, v = lookup["baseline", item], lookup["v4", item]
            summary[field + "s"].append({field: item, "baseline": b, "v4": v,
                                        "mse_reduction_pct": 100 * (1 - v["mse"] / b["mse"])})
    blocks = {(r["model"], r["shot"], r["group"]): r for r in block_rows if r["epoch"] == 1000}
    summary["epoch1000_block_mse_wins"] = sum(blocks["v4", s, g]["mse"] < blocks["baseline", s, g]["mse"]
                                              for s in shots for g in range(1, 51))
    summary["zero_prediction_mse"] = float(np.mean([np.mean(r.astype(np.float64) ** 2) for r in references.values()]))
    summary["epoch1000_energy_decomposition"] = energy_decomposition(shots, references, reconstructions)
    summary["environment"] = {"numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"audit": audit, "epoch1000": summary["epochs"][-1], "block_wins": summary["epoch1000_block_mse_wins"]}), flush=True)
    plot_curves(overall, shot_rows, group_rows, block_rows, shots)
    plot_waveforms(shots, references, reconstructions, groups)
    write_gallery(shots, groups)
    print(f"Results: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
