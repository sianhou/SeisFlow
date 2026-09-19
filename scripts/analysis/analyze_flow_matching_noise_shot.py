#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def patch_complexity(patches):
    vertical = np.mean(np.abs(np.diff(patches, axis=1)), axis=(1, 2))
    horizontal = np.mean(np.abs(np.diff(patches, axis=2)), axis=(1, 2))
    return vertical + horizontal


def load_training_statistics(train_dir):
    complexity = []
    energy = []
    for path in sorted(train_dir.glob("*.npy")):
        patches = np.load(path, mmap_mode="r")
        complexity.append(patch_complexity(patches))
        energy.append(np.mean(patches * patches, axis=(1, 2)))
    return np.concatenate(complexity), np.concatenate(energy)


def parse_metric(path, shot_name):
    pattern = re.compile(rf"{re.escape(shot_name)}:\s*([-+0-9.eE]+)")
    for line in path.read_text().splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            return float(match.group(1))
    raise KeyError(f"{shot_name} not found in {path}")


def parse_metric_mean(path):
    values = []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"shot_\d+:\s*([-+0-9.eE]+)", line.strip())
        if match:
            values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"No shot metrics found in {path}")
    return float(np.mean(values))


def load_epoch_metrics(result_dir, shot_name):
    records = []
    for directory in sorted(result_dir.glob("diff_recon_shot_epoch_*")):
        epoch = int(directory.name.rsplit("_", 1)[-1])
        records.append(
            (
                epoch,
                parse_metric(directory / "psnr.txt", shot_name),
                parse_metric_mean(directory / "psnr.txt"),
                parse_metric(directory / "ssim.txt", shot_name),
                parse_metric_mean(directory / "ssim.txt"),
            )
        )
    return np.asarray(records)


def plot_flow_path(shot, patch, position, output_path, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(patch.shape).astype(np.float32)
    times = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

    figure = plt.figure(figsize=(18, 7.5))
    grid = figure.add_gridspec(2, 6, height_ratios=(1.05, 1.0))
    shot_axis = figure.add_subplot(grid[0, :])
    shot_axis.imshow(shot, cmap="seismic", vmin=-2, vmax=2, aspect="auto")
    row, column = position
    shot_axis.add_patch(
        Rectangle((column, row), 64, 64, fill=False, edgecolor="#ffcc00", linewidth=2)
    )
    shot_axis.set_title(
        f"Worst validation shot and selected highest-complexity patch at "
        f"(row={row}, column={column})"
    )
    shot_axis.set_xlabel("Trace/sample axis")
    shot_axis.set_ylabel("Receiver/time axis")

    for axis, time_value in zip(
        [figure.add_subplot(grid[1, index]) for index in range(6)],
        times,
    ):
        noisy_patch = (1.0 - time_value) * noise + time_value * patch
        axis.imshow(noisy_patch, cmap="seismic", vmin=-2, vmax=2, aspect="equal")
        correlation = (
            np.nan
            if time_value == 0.0
            else np.corrcoef(noisy_patch.ravel(), patch.ravel())[0, 1]
        )
        correlation_text = "n/a" if np.isnan(correlation) else f"{correlation:.2f}"
        axis.set_title(f"t={time_value:g}\ncorr(clean)={correlation_text}")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        r"Linear flow-matching path: $x_t=(1-t)\epsilon+t x_1$",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(
        shot_patches,
        train_complexity,
        train_energy,
        complex_index,
        energy_index,
        epoch_metrics,
        output_path,
        seed,
):
    shot_complexity = patch_complexity(shot_patches)
    shot_energy = np.mean(shot_patches * shot_patches, axis=(1, 2))
    selected = {
        "Highest complexity": shot_patches[complex_index],
        "Highest energy": shot_patches[energy_index],
        "All shot patches": shot_patches.reshape(-1),
    }

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(
        train_complexity,
        bins=80,
        density=True,
        alpha=0.65,
        color="#3066be",
        label="Training patches",
    )
    axes[0, 0].hist(
        shot_complexity,
        bins=35,
        density=True,
        histtype="step",
        linewidth=2,
        color="#d1495b",
        label="shot_0127 patches",
    )
    axes[0, 0].axvline(
        shot_complexity[complex_index], color="#ff8c00", linewidth=2,
        label=f"max patch: train p{np.mean(train_complexity <= shot_complexity[complex_index]) * 100:.2f}",
    )
    axes[0, 0].set_title("Patch structural-complexity distribution")
    axes[0, 0].set_xlabel("Mean absolute gradients (vertical + horizontal)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hist(
        train_energy,
        bins=80,
        density=True,
        alpha=0.65,
        color="#3066be",
        label="Training patches",
    )
    axes[0, 1].hist(
        shot_energy,
        bins=35,
        density=True,
        histtype="step",
        linewidth=2,
        color="#d1495b",
        label="shot_0127 patches",
    )
    axes[0, 1].axvline(
        shot_energy[energy_index], color="#ff8c00", linewidth=2,
        label=f"max patch: train p{np.mean(train_energy <= shot_energy[energy_index]) * 100:.2f}",
    )
    axes[0, 1].set_title("Patch energy distribution")
    axes[0, 1].set_xlabel("Mean squared amplitude")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend(frameon=False)

    rng = np.random.default_rng(seed)
    times = np.linspace(0.001, 0.999, 300)
    for label, values in selected.items():
        centered = values - np.mean(values)
        signal_std = float(np.std(centered))
        noise = rng.standard_normal(values.shape)
        noise_std = float(np.std(noise))
        snr_db = 20.0 * np.log10(
            times * signal_std / ((1.0 - times) * noise_std)
        )
        axes[1, 0].plot(times, snr_db, linewidth=2, label=label)
        crossover = noise_std / (noise_std + signal_std)
        axes[1, 0].axvline(crossover, linewidth=1, linestyle="--", alpha=0.35)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylim(-40, 40)
    axes[1, 0].set_title("Clean-component SNR along the flow path")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("20 log10(signal RMS / noise RMS), dB")
    axes[1, 0].legend(frameon=False)

    epochs = epoch_metrics[:, 0]
    psnr_axis = axes[1, 1]
    psnr_axis.plot(
        epochs, epoch_metrics[:, 1], marker="o", color="#d1495b",
        label="shot_0127 PSNR",
    )
    psnr_axis.plot(
        epochs, epoch_metrics[:, 2], marker="o", color="#3066be",
        label="75-shot mean PSNR",
    )
    psnr_axis.set_xlabel("Epoch")
    psnr_axis.set_ylabel("PSNR (dB)")
    psnr_axis.set_title("Reconstruction trajectory")
    ssim_axis = psnr_axis.twinx()
    ssim_axis.plot(
        epochs, epoch_metrics[:, 3], linestyle="--", color="#d1495b",
        label="shot_0127 SSIM",
    )
    ssim_axis.set_ylabel("shot_0127 SSIM")
    handles1, labels1 = psnr_axis.get_legend_handles_labels()
    handles2, labels2 = ssim_axis.get_legend_handles_labels()
    psnr_axis.legend(handles1 + handles2, labels1 + labels2, frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Why shot_0127 remains difficult", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_checkpoint_loss(
        checkpoint_root,
        checkpoint_epochs,
        shot_name,
        shot_patches,
        conditioning,
        complexity,
        output_dir,
        seed,
):
    import sys

    import torch

    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from models.wrapper import DiTTransformer2DWrapper

    times = np.asarray((0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95))
    order = np.argsort(complexity)
    group_size = max(len(order) // 10, 1)
    typical_start = (len(order) - group_size) // 2
    groups = {
        "typical": order[typical_start:typical_start + group_size],
        "complex": order[-group_size:],
    }

    prepared = {}
    for group_index, (group_name, indices) in enumerate(groups.items()):
        clean = torch.from_numpy(np.array(shot_patches[indices], copy=True))[:, None]
        cond = torch.from_numpy(np.array(conditioning[indices], copy=True))
        generator = torch.Generator().manual_seed(seed + group_index)
        group_samples = []
        for time_value in times:
            noise = torch.randn(clean.shape, generator=generator)
            x_t = (1.0 - time_value) * noise + time_value * clean
            target = clean - noise
            group_samples.append((float(time_value), x_t, target, cond))
        prepared[group_name] = group_samples

    records = []
    for epoch in checkpoint_epochs:
        checkpoint = checkpoint_root / f"checkpoint_epoch_{epoch:05d}"
        model = DiTTransformer2DWrapper.from_pretrained(
            checkpoint,
            device="cpu",
            use_ema=True,
        ).eval()
        with torch.inference_mode():
            for group_name, samples in prepared.items():
                for time_value, x_t, target, cond in samples:
                    time_batch = torch.full((len(x_t),), time_value)
                    prediction = model(
                        x_t,
                        time_batch,
                        extra={"concat_conditioning": cond},
                    )
                    mse = torch.mean((prediction.float() - target.float()) ** 2)
                    records.append((epoch, group_name, time_value, float(mse)))
        del model

    csv_path = output_dir / f"{shot_name}_t_binned_velocity_mse.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("epoch", "complexity_group", "t", "velocity_mse"))
        writer.writerows(records)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    color_map = plt.get_cmap("viridis")
    for axis, group_name in zip(axes, ("typical", "complex")):
        for time_index, time_value in enumerate(times):
            rows = [
                row for row in records
                if row[1] == group_name and np.isclose(row[2], time_value)
            ]
            axis.plot(
                [row[0] for row in rows],
                [row[3] for row in rows],
                marker="o",
                color=color_map(time_index / (len(times) - 1)),
                label=f"t={time_value:g}",
            )
        axis.set_title(f"{group_name.capitalize()} patches ({group_size})")
        axis.set_xlabel("Checkpoint epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("EMA velocity-target MSE")
    axes[1].legend(frameon=False, ncol=2)
    figure.suptitle("DiT checkpoint error by flow time and patch complexity", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        output_dir / f"{shot_name}_t_binned_checkpoint_loss.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--shot_id", default="0127")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--checkpoint_root", type=Path)
    parser.add_argument(
        "--checkpoint_epochs",
        nargs="+",
        type=int,
        default=(100, 500, 1000, 1500, 2000),
    )
    args = parser.parse_args()

    shot_name = f"shot_{args.shot_id}"
    patch_name = f"patches_{args.shot_id}"
    shot = np.load(args.dataset_dir / "shot" / f"{shot_name}.npy")
    shot_patches = np.load(args.dataset_dir / "valid" / f"{patch_name}.npy")
    conditioning = np.load(
        args.dataset_dir / "valid_dim" / f"{patch_name}.npy"
    )
    metadata = np.load(args.dataset_dir / "valid_aux" / f"{patch_name}.npz")
    positions = metadata["positions"]

    complexity = patch_complexity(shot_patches)
    energy = np.mean(shot_patches * shot_patches, axis=(1, 2))
    complex_index = int(np.argmax(complexity))
    energy_index = int(np.argmax(energy))
    train_complexity, train_energy = load_training_statistics(
        args.dataset_dir / "train"
    )
    epoch_metrics = load_epoch_metrics(args.result_dir, shot_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_flow_path(
        shot,
        shot_patches[complex_index],
        positions[complex_index],
        args.output_dir / f"{shot_name}_flow_matching_path.png",
        args.seed,
    )
    plot_diagnostics(
        shot_patches,
        train_complexity,
        train_energy,
        complex_index,
        energy_index,
        epoch_metrics,
        args.output_dir / f"{shot_name}_noise_and_training_diagnostics.png",
        args.seed,
    )
    if args.checkpoint_root is not None:
        evaluate_checkpoint_loss(
            args.checkpoint_root,
            args.checkpoint_epochs,
            shot_name,
            shot_patches,
            conditioning,
            complexity,
            args.output_dir,
            args.seed,
        )

    complex_percentile = np.mean(train_complexity <= complexity[complex_index]) * 100
    energy_percentile = np.mean(train_energy <= energy[energy_index]) * 100
    print(f"shot={shot_name} patches={len(shot_patches)}")
    print(
        f"highest_complexity_patch={complex_index} "
        f"position={positions[complex_index].tolist()} "
        f"train_percentile={complex_percentile:.4f}"
    )
    print(
        f"highest_energy_patch={energy_index} "
        f"position={positions[energy_index].tolist()} "
        f"train_percentile={energy_percentile:.4f}"
    )
    for label, index in (("complex", complex_index), ("energy", energy_index)):
        signal_std = float(np.std(shot_patches[index]))
        crossover = 1.0 / (1.0 + signal_std)
        print(
            f"{label}_patch_std={signal_std:.6f} "
            f"zero_db_snr_t={crossover:.6f} "
            f"uniform_t_fraction_above={1.0 - crossover:.6f}"
        )


if __name__ == "__main__":
    main()
