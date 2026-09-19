#!/usr/bin/env python3
"""Locate large-error regions and analyze their frequency/wavelet content.

The current comparison directories retain DiffShot PNGs and scalar metrics, but
not reconstructed NPY arrays.  Therefore the residual maps extracted here are
display-resolution proxies.  Their global energy is calibrated with mse.txt;
all signal frequency and wavelet statistics are computed from the exact NPY.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr


AXIS_BOX = (2287.02 / 3240.0, 79.1 / 900.0, 3062.34 / 3240.0, 817.0 / 900.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, help="Exact original 2-D shot NPY.")
    parser.add_argument(
        "--error-png", action="append", required=True, metavar="NAME=PNG",
        help="DiffShot three-panel PNG; repeat once per model.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", nargs=2, type=int, default=(64, 128), metavar=("ROWS", "COLS"))
    parser.add_argument("--regions", type=int, default=4)
    return parser.parse_args()


def parse_specs(items):
    specs = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PNG, got {item!r}")
        name, path = item.split("=", 1)
        specs.append((name, Path(path)))
    return specs


def read_mse(png_path, shot_stem):
    text = (png_path.parent / "mse.txt").read_text()
    match = re.search(rf"^{re.escape(shot_stem)}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"No {shot_stem} entry in {png_path.parent / 'mse.txt'}")
    return float(match.group(1))


def invert_diffshot_panel(png_path, output_shape, value_min, value_max):
    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = AXIS_BOX
    crop = rgb[
        max(0, round(y0 * height) + 1):min(height, round(y1 * height) - 1),
        max(0, round(x0 * width) + 1):min(width, round(x1 * width) - 1),
    ]
    lut_size = 8193
    lut = plt.get_cmap("seismic")(np.linspace(0.0, 1.0, lut_size))[:, :3]
    indices = cKDTree(lut).query(crop.reshape(-1, 3), workers=-1)[1]
    values = value_min + indices.reshape(crop.shape[:2]) * (value_max - value_min) / (lut_size - 1)
    # Displayed data are clockwise-rotated: (rows, cols) -> (cols, rows).
    rotated_shape = (output_shape[1], output_shape[0])
    resized = np.asarray(
        Image.fromarray(values.astype(np.float32), mode="F").resize(
            (rotated_shape[1], rotated_shape[0]), Image.Resampling.BILINEAR
        ), dtype=np.float32,
    )
    return np.rot90(resized, k=1)


def calibrate_residual(proxy, true_mse):
    proxy_mse = float(np.mean(proxy.astype(np.float64) ** 2))
    return proxy * np.sqrt(true_mse / max(proxy_mse, 1e-12))


def candidate_windows(score, shape, stride=None):
    wh, ww = shape
    sh, sw = stride or (wh // 2, ww // 2)
    rows = list(range(0, max(score.shape[0] - wh + 1, 1), sh))
    cols = list(range(0, max(score.shape[1] - ww + 1, 1), sw))
    if not rows or rows[-1] != score.shape[0] - wh:
        rows.append(score.shape[0] - wh)
    if not cols or cols[-1] != score.shape[1] - ww:
        cols.append(score.shape[1] - ww)
    result = []
    for row in rows:
        for col in cols:
            value = float(np.mean(score[row:row + wh, col:col + ww]))
            result.append((value, row, col, wh, ww))
    return result


def overlap_ratio(a, b):
    _, ar, ac, ah, aw = a
    _, br, bc, bh, bw = b
    ih = max(0, min(ar + ah, br + bh) - max(ar, br))
    iw = max(0, min(ac + aw, bc + bw) - max(ac, bc))
    return (ih * iw) / float(min(ah * aw, bh * bw))


def select_regions(candidates, count, reverse=True, forbidden=()):
    selected = []
    for item in sorted(candidates, reverse=reverse):
        if any(overlap_ratio(item, old) > 0.15 for old in selected):
            continue
        if any(overlap_ratio(item, old) > 0.15 for old in forbidden):
            continue
        selected.append(item)
        if len(selected) == count:
            break
    return selected


def spectrum_1d(block, axis):
    data = block - np.mean(block, axis=axis, keepdims=True)
    n = data.shape[axis]
    window_shape = [1] * data.ndim
    window_shape[axis] = n
    window = np.hanning(n).reshape(window_shape)
    spec = np.fft.rfft(data * window, axis=axis)
    power = np.mean(np.abs(spec) ** 2, axis=1 - axis)
    freq = np.fft.rfftfreq(n)
    power = power / max(float(np.sum(power)), 1e-15)
    return freq / 0.5, power


def spectral_stats(block, axis):
    freq, power = spectrum_1d(block, axis)
    centroid = float(np.sum(freq * power))
    low = float(np.sum(power[freq < 0.25]))
    mid = float(np.sum(power[(freq >= 0.25) & (freq < 0.5)]))
    high = float(np.sum(power[freq >= 0.5]))
    return centroid, low, mid, high


def haar_levels(block, levels=3):
    current = block.astype(np.float64)
    result = []
    total = float(np.sum(current ** 2))
    for _ in range(levels):
        h, w = current.shape
        current = current[:h - h % 2, :w - w % 2]
        a, b = current[0::2, 0::2], current[0::2, 1::2]
        c, d = current[1::2, 0::2], current[1::2, 1::2]
        ll = (a + b + c + d) / 2.0
        lh = (a - b + c - d) / 2.0
        hl = (a + b - c - d) / 2.0
        hh = (a - b - c + d) / 2.0
        detail = float(np.sum(lh ** 2) + np.sum(hl ** 2) + np.sum(hh ** 2))
        result.append((ll, lh, hl, hh, detail / max(total, 1e-15)))
        current = ll
    return result


def temporal_bands(shot):
    centered = shot - np.mean(shot, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    freq = np.fft.rfftfreq(shot.shape[1]) / 0.5
    outputs = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 1.00001)):
        mask = (freq >= lo) & (freq < hi)
        outputs.append(np.fft.irfft(spectrum * mask[None, :], n=shot.shape[1], axis=1).real)
    return outputs


def add_boxes(axis, regions, colors=None):
    from matplotlib.patches import Rectangle
    for index, (_, row, col, height, width) in enumerate(regions, 1):
        color = colors[index - 1] if colors else "lime"
        axis.add_patch(Rectangle((col, row), width, height, fill=False, lw=1.5, ec=color))
        axis.text(col + 4, row + 14, f"R{index}", color=color, fontsize=8, weight="bold")


def partial_spearman(x, y, control):
    x_rank, y_rank, z_rank = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack((np.ones_like(z_rank), z_rank))
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shot_path = Path(args.shot)
    shot = np.load(shot_path).astype(np.float32)
    if shot.ndim != 2:
        raise ValueError(f"Expected 2-D shot, got {shot.shape}")
    specs = parse_specs(args.error_png)

    residuals, true_mses = {}, {}
    for name, png_path in specs:
        mse = read_mse(png_path, shot_path.stem)
        proxy = invert_diffshot_panel(png_path, shot.shape, float(shot.min()), float(shot.max()))
        residuals[name] = calibrate_residual(proxy, mse)
        true_mses[name] = mse

    normalized = []
    for residual in residuals.values():
        squared = residual ** 2
        normalized.append(squared / max(float(np.percentile(squared, 95)), 1e-12))
    consensus = np.mean(normalized, axis=0)
    candidates = candidate_windows(consensus, tuple(args.window))
    high_regions = select_regions(candidates, args.regions, reverse=True)

    # Controls come from the lowest-error third, then minimize the RMS mismatch.
    # A large remaining mismatch is itself evidence that absolute error is energy-driven.
    low_pool = sorted(candidates)[:max(40, len(candidates) // 3)]
    control_regions = []
    for high in high_regions:
        _, hr, hc, hh, hw = high
        target_rms = float(np.sqrt(np.mean(shot[hr:hr + hh, hc:hc + hw] ** 2)))
        choices = []
        for low in low_pool:
            if any(overlap_ratio(low, x) > 0.15 for x in high_regions + control_regions):
                continue
            _, row, col, height, width = low
            rms = float(np.sqrt(np.mean(shot[row:row + height, col:col + width] ** 2)))
            choices.append((abs(rms - target_rms), low))
        if choices:
            control_regions.append(min(choices, key=lambda x: x[0])[1])

    colors = ["#00ff66", "#ffd43b", "#39d0ff", "#ff7ad9", "#ff914d", "#b197fc"]
    extent = (0, shot.shape[1], shot.shape[0], 0)
    vmax = max(abs(float(shot.min())), abs(float(shot.max())))

    fig, axes = plt.subplots(2, 2, figsize=(16, 7.5), constrained_layout=True)
    panels = [("Exact original", shot)] + [(f"{name} residual proxy", value) for name, value in residuals.items()]
    for axis, (title, data) in zip(axes.flat, panels):
        limit = vmax if title == "Exact original" else float(np.percentile(np.abs(data), 99.5))
        image = axis.imshow(data, cmap="seismic", vmin=-limit, vmax=limit, aspect="auto", extent=extent)
        add_boxes(axis, high_regions, colors)
        axis.set_title(title)
        axis.set_xlabel("sample / time axis")
        axis.set_ylabel("receiver / spatial axis")
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.savefig(output_dir / "shot_0127_error_regions.png", dpi=180)
    plt.close(fig)

    bands = temporal_bands(shot)
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), constrained_layout=True, sharex=True, sharey=True)
    band_titles = ("Original", "Low: 0-0.25 Nyquist", "Mid: 0.25-0.50 Nyquist", "High: 0.50-1.00 Nyquist")
    for axis, title, data in zip(axes, band_titles, [shot] + bands):
        limit = float(np.percentile(np.abs(data), 99.5))
        image = axis.imshow(data, cmap="seismic", vmin=-limit, vmax=limit, aspect="auto", extent=extent)
        add_boxes(axis, high_regions, colors)
        axis.set_title(title)
        axis.set_ylabel("receiver")
        fig.colorbar(image, ax=axis, fraction=0.012, pad=0.01)
    axes[-1].set_xlabel("sample / time axis")
    fig.savefig(output_dir / "shot_0127_temporal_frequency_bands.png", dpi=180)
    plt.close(fig)

    rows = []
    for kind, regions in (("high_error", high_regions), ("control", control_regions)):
        for index, (_, row, col, height, width) in enumerate(regions, 1):
            block = shot[row:row + height, col:col + width]
            time_stats = spectral_stats(block, axis=1)
            space_stats = spectral_stats(block, axis=0)
            wavelets = haar_levels(block)
            record = {
                "kind": kind, "region": index, "row_start": row, "row_end": row + height,
                "col_start": col, "col_end": col + width,
                "signal_rms": float(np.sqrt(np.mean(block ** 2))),
                "time_centroid_nyquist": time_stats[0], "time_low_energy": time_stats[1],
                "time_mid_energy": time_stats[2], "time_high_energy": time_stats[3],
                "space_centroid_nyquist": space_stats[0], "space_low_energy": space_stats[1],
                "space_mid_energy": space_stats[2], "space_high_energy": space_stats[3],
            }
            for level, wavelet in enumerate(wavelets, 1):
                record[f"haar_level{level}_detail_energy"] = wavelet[4]
            for name, residual in residuals.items():
                error = residual[row:row + height, col:col + width]
                record[f"{name}_proxy_mse"] = float(np.mean(error ** 2))
                record[f"{name}_proxy_time_high_energy"] = spectral_stats(error, axis=1)[3]
            rows.append(record)

    with (output_dir / "shot_0127_region_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    window_rows = []
    for score, row, col, height, width in candidates:
        block = shot[row:row + height, col:col + width]
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms < 0.05:
            continue
        time_stats = spectral_stats(block, axis=1)
        space_stats = spectral_stats(block, axis=0)
        wavelet = haar_levels(block, levels=1)[0][4]
        window_rows.append({
            "consensus_error": score, "signal_rms": rms,
            "time_high_energy": time_stats[3], "space_high_energy": space_stats[3],
            "haar_level1_detail_energy": wavelet,
        })
    correlations = {}
    partial_correlations = {}
    error_values = [row["consensus_error"] for row in window_rows]
    rms_values = [row["signal_rms"] for row in window_rows]
    for key in ("signal_rms", "time_high_energy", "space_high_energy", "haar_level1_detail_energy"):
        feature_values = [row[key] for row in window_rows]
        correlations[key] = float(spearmanr(error_values, feature_values).statistic)
        partial_correlations[key] = (
            float("nan") if key == "signal_rms"
            else partial_spearman(error_values, feature_values, rms_values)
        )
    with (output_dir / "shot_0127_window_correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "feature", "spearman_error_correlation", "partial_spearman_controlling_rms"
        ])
        writer.writeheader()
        writer.writerows(
            {"feature": key, "spearman_error_correlation": value,
             "partial_spearman_controlling_rms": partial_correlations[key]}
            for key, value in correlations.items()
        )

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    labels = {
        "signal_rms": "signal RMS", "time_high_energy": "temporal high-band fraction",
        "space_high_energy": "spatial high-band fraction",
        "haar_level1_detail_energy": "Haar L1 detail fraction",
    }
    y = np.asarray([row["consensus_error"] for row in window_rows])
    for axis, key in zip(axes, labels):
        x = np.asarray([row[key] for row in window_rows])
        axis.scatter(x, y, s=13, alpha=0.55, color="#374fc7")
        axis.set_xlabel(labels[key])
        axis.set_ylabel("consensus residual score")
        partial_text = "" if key == "signal_rms" else f"; partial = {partial_correlations[key]:.3f}"
        axis.set_title(f"Spearman r = {correlations[key]:.3f}{partial_text}")
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "shot_0127_error_feature_correlations.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, direction, dim in ((axes[0], "Temporal/sample axis", 1), (axes[1], "Spatial/receiver axis", 0)):
        for kind, regions, color, linestyle in (
            ("high-error", high_regions, "#d62728", "-"), ("control", control_regions, "#1f77b4", "--")
        ):
            spectra = []
            for _, row, col, height, width in regions:
                freq, power = spectrum_1d(shot[row:row + height, col:col + width], axis=dim)
                spectra.append(power)
            if spectra:
                mean = np.mean(spectra, axis=0)
                axis.plot(freq, mean, color=color, ls=linestyle, lw=2, label=kind)
                axis.fill_between(freq, 0, mean, color=color, alpha=0.12)
        axis.axvspan(0.5, 1.0, color="grey", alpha=0.1, label="high band" if dim == 1 else None)
        axis.set_yscale("log")
        axis.set_title(direction)
        axis.set_xlabel("normalized frequency (1 = Nyquist)")
        axis.set_ylabel("normalized power")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output_dir / "shot_0127_region_spectra.png", dpi=180)
    plt.close(fig)

    show_count = min(3, len(high_regions))
    fig, axes = plt.subplots(show_count, 5, figsize=(14, 3.1 * show_count), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for index, region in enumerate(high_regions[:show_count]):
        _, row, col, height, width = region
        block = shot[row:row + height, col:col + width]
        ll, lh, hl, hh, _ = haar_levels(block, levels=1)[0]
        for axis, title, data in zip(axes[index], (f"R{index + 1} signal", "H detail", "V detail", "D detail", "error consensus"),
                                     (block, lh, hl, hh, consensus[row:row + height, col:col + width])):
            cmap = "magma" if title == "error consensus" else "seismic"
            limit = float(np.percentile(np.abs(data), 99.0))
            if cmap == "seismic":
                axis.imshow(data, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            else:
                axis.imshow(data, cmap=cmap, aspect="auto")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.savefig(output_dir / "shot_0127_haar_details.png", dpi=180)
    plt.close(fig)

    high_rows = [row for row in rows if row["kind"] == "high_error"]
    control_rows = [row for row in rows if row["kind"] == "control"]
    def avg(group, key):
        return float(np.mean([row[key] for row in group])) if group else float("nan")
    lines = [
        "# shot_0127 frequency/wavelet summary", "",
        "Residual locations are approximate because they are recovered from rendered DiffShot PNGs; "
        "original-signal frequency and Haar statistics use the exact NPY.", "",
        f"- High-error windows: {len(high_rows)}; lowest-error controls: {len(control_rows)}.",
        f"- Mean signal RMS: high-error {avg(high_rows, 'signal_rms'):.4f}, "
        f"control {avg(control_rows, 'signal_rms'):.4f}; exact energy matching was not possible in the low-error pool.",
        f"- Temporal high-band energy (0.5-1.0 Nyquist): high-error {avg(high_rows, 'time_high_energy'):.4f}, "
        f"control {avg(control_rows, 'time_high_energy'):.4f}.",
        f"- Spatial high-band energy (0.5-1.0 Nyquist): high-error {avg(high_rows, 'space_high_energy'):.4f}, "
        f"control {avg(control_rows, 'space_high_energy'):.4f}.",
        f"- Haar level-1 detail energy: high-error {avg(high_rows, 'haar_level1_detail_energy'):.4f}, "
        f"control {avg(control_rows, 'haar_level1_detail_energy'):.4f}.", "",
        "## All active windows: Spearman correlation with residual score", "",
        f"- Signal RMS: {correlations['signal_rms']:.4f}.",
        f"- Temporal high-band fraction: {correlations['time_high_energy']:.4f}.",
        f"- Spatial high-band fraction: {correlations['space_high_energy']:.4f}.",
        f"- Haar level-1 detail fraction: {correlations['haar_level1_detail_energy']:.4f}.", "",
        "After controlling for signal RMS (partial Spearman):", "",
        f"- Temporal high-band fraction: {partial_correlations['time_high_energy']:.4f}.",
        f"- Spatial high-band fraction: {partial_correlations['space_high_energy']:.4f}.",
        f"- Haar level-1 detail fraction: {partial_correlations['haar_level1_detail_energy']:.4f}.", "",
        "## Regions", "",
    ]
    for row in high_rows:
        lines.append(
            f"- R{row['region']}: receiver [{row['row_start']}, {row['row_end']}), "
            f"sample [{row['col_start']}, {row['col_end']}), RMS {row['signal_rms']:.4f}, "
            f"temporal high {row['time_high_energy']:.4f}, spatial high {row['space_high_energy']:.4f}."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
