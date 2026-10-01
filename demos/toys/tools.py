import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.display_functions import clear_output, display
from ipywidgets import widgets
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 提高 CUDA 运算的可复现性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_ddpm_coefficients(betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_alphas_cumprod,
                           sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    # 每个子图：(标题, 纵轴名称, 曲线列表)
    # 每条曲线：(张量, 图例, 颜色, 线型)
    panels = [
        (
            "Forward and posterior variance",
            "Variance",
            [
                (betas, r"$\beta_t$", "#3B82F6", "-"),
                (posterior_variance, r"$\tilde{\beta}_t$", "#EF6461", "--"),
            ],
        ),
        (
            "Single-step signal retention",
            "Coefficient",
            [
                (alphas, r"$\alpha_t$", "#10B981", "-"),
            ],
        ),
        (
            "Reverse-step scaling",
            "Coefficient",
            [
                (sqrt_recip_alphas, r"$1/\sqrt{\alpha_t}$", "#8B5CF6", "-"),
            ],
        ),
        (
            "Cumulative signal retention",
            "Coefficient",
            [
                (alphas_cumprod, r"$\bar{\alpha}_t$", "#3B82F6", "-"),
                (
                    alphas_cumprod_prev,
                    r"$\bar{\alpha}_{t-1}$",
                    "#F59E0B",
                    "--",
                ),
            ],
        ),
        (
            "Signal amplitude",
            "Amplitude",
            [
                (sqrt_alphas_cumprod, r"$\sqrt{\bar{\alpha}_t}$", "#0891B2", "-"),
            ],
        ),
        (
            "Noise amplitude",
            "Amplitude",
            [
                (
                    sqrt_one_minus_alphas_cumprod,
                    r"$\sqrt{1-\bar{\alpha}_t}$",
                    "#EF6461",
                    "-",
                ),
            ],
        ),
    ]

    # 数组索引 0 对应数学时间步 t=1
    timesteps = np.arange(1, betas.numel() + 1)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 8),
        dpi=150,
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    fig.patch.set_facecolor("white")

    for ax, (title, ylabel, curves) in zip(axes.flat, panels):
        ax.set_facecolor("#FAFBFD")

        for values, label, color, linestyle in curves:
            ax.plot(
                timesteps,
                values.detach().cpu().numpy(),
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
            )

        ax.set_title(title, fontsize=12, color="#1F2937", pad=12)
        ax.set_xlabel(r"Diffusion step $t$", fontsize=10, color="#64748B")
        ax.set_ylabel(ylabel, fontsize=10, color="#64748B")
        ax.set_xlim(1, timesteps[-1])
        ax.margins(y=0.08)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(
            axis="both",
            labelsize=9,
            colors="#64748B",
            length=0,
            pad=6,
        )

        ax.set_axisbelow(True)
        ax.grid(color="#E2E8F0", linewidth=0.7, alpha=0.8)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.legend(
            loc="best",
            fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
            framealpha=0.95,
        )

    fig.suptitle(
        "DDPM Diffusion Coefficients",
        fontsize=18,
        fontweight="semibold",
        color="#0F172A",
    )

    plt.show()


def plot_trajectory(
        x_T,
        x_0,
        trajectory=None,
        trajectory_interval=50,
        title=None,
        xlim=(-4, 4),
        ylim=(-4, 4),
):
    """绘制二维 DDPM 起点、终点及可选采样轨迹。

    trajectory: 形状为 (T + 1, n_samples, 2)，顺序为 x_T → x_0。
    trajectory_interval: 轨迹每隔多少步取一个点，始终保留终点。
    """

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    x_T = to_numpy(x_T)
    x_0 = to_numpy(x_0)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFD")

    if trajectory is not None:
        if (
                not isinstance(trajectory_interval, (int, np.integer))
                or trajectory_interval < 1
        ):
            raise ValueError("trajectory_interval 必须为正整数")

        paths = to_numpy(trajectory)
        n_steps = paths.shape[0] - 1

        indices = np.arange(0, n_steps + 1, trajectory_interval)
        if indices[-1] != n_steps:
            indices = np.append(indices, n_steps)

        segments = paths[indices, :, :2].transpose(1, 0, 2)
        ax.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=0.6,
                alpha=0.18,
                zorder=2,
            )
        )

    ax.scatter(
        x_T[:, 0], x_T[:, 1],
        s=18, color="#EF6461", alpha=0.75,
        edgecolors="none", label=r"Initial $x_T$", zorder=3,
    )
    ax.scatter(
        x_0[:, 0], x_0[:, 1],
        s=20, color="#3B82F6", alpha=0.85,
        edgecolors="none", label=r"Generated $x_0$", zorder=4,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(color="#E5E7EB", linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(length=0, labelsize=10, colors="#778195")
    ax.legend(
        loc="upper right",
        facecolor="white",
        edgecolor="#E5E7EB",
        framealpha=0.95,
        fontsize=10,
        markerscale=1.5,
    )

    if title is not None:
        ax.set_title(title, fontsize=14, pad=16)

    fig.tight_layout()
    return fig, ax


def view_ddpm_trajectory(
        trajectory,
        trajectory_interval=10,
        xlim=(-4, 4),
        ylim=(-4, 4),
):
    """交互查看 DDPM 的采样轨迹。

    Args:
        trajectory: Tensor 或 ndarray，形状为 (T + 1, N, 2)，
            顺序为 x_T、x_{T-1}、...、x_0。
        trajectory_interval: 路径连线的时间步间隔，始终包含当前步。
            不影响滑条查看每一步的位置。
        xlim: 横坐标显示范围。
        ylim: 纵坐标显示范围。

    Returns:
        包含滑条、路径开关和图像的 Jupyter 控件。
    """
    if isinstance(trajectory, torch.Tensor):
        paths = trajectory.detach().cpu().numpy()
    else:
        paths = np.asarray(trajectory)

    total_steps = paths.shape[0] - 1

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=total_steps,
        step=1,
        description="采样步数",
        continuous_update=True,
        layout=widgets.Layout(width="650px"),
    )
    show_paths = widgets.Checkbox(
        value=True,
        description="显示历史路径",
        indent=False,
    )
    output = widgets.Output()

    # 创建一次图像，拖动时更新已有对象
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFD")

    lines = LineCollection(
        [],
        colors="black",
        linewidths=0.6,
        alpha=0.15,
        zorder=2,
    )
    ax.add_collection(lines)

    ax.scatter(
        paths[0, :, 0], paths[0, :, 1],
        s=14, color="#EF6461", alpha=0.35,
        edgecolors="none", label=r"Initial $x_T$", zorder=3,
    )
    current = ax.scatter(
        paths[0, :, 0], paths[0, :, 1],
        s=20, color="#3B82F6", alpha=0.85,
        edgecolors="none", label="Current samples", zorder=4,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(color="#E2E8F0", linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(length=0, colors="#64748B")
    ax.legend(
        loc="upper right",
        facecolor="white",
        edgecolor="#E2E8F0",
    )
    title = ax.set_title("", fontsize=13, pad=14)
    fig.tight_layout()

    def update(change=None):
        """根据滑条位置更新散点和截至当前步的历史路径。"""
        step = slider.value
        current.set_offsets(paths[step, :, :2])

        if show_paths.value and step > 0:
            indices = np.arange(0, step + 1, trajectory_interval)
            if indices[-1] != step:
                indices = np.append(indices, step)

            segments = paths[indices, :, :2].transpose(1, 0, 2)
            lines.set_segments(segments)
        else:
            lines.set_segments([])

        title.set_text(
            f"DDPM: step {step}/{total_steps}  "
            f"—  $x_{{{total_steps - step}}}$"
        )

        with output:
            clear_output(wait=True)
            display(fig)

    slider.observe(update, names="value")
    show_paths.observe(update, names="value")

    update()
    plt.close(fig)

    return widgets.VBox([slider, show_paths, output])
