import numpy as np
from matplotlib import pyplot as plt


def plot_seismic_row(imgs, fig_name, title="", vmin=-1, vmax=1, ):
    num_figs = imgs.shape[0]

    fig, axes = plt.subplots(1, num_figs, figsize=(6, 12))
    axes = axes.ravel()

    for i in range(num_figs):
        axes[i].imshow(imgs[i].T, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{i}")
        axes[i].axis("off")

    plt.tight_layout()
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)
    plt.close()


def plot_seismic_grid(
    imgs,
    fig_name,
    title="",
    vmin=None,
    vmax=None,
    size=2,
    cmap="seismic",
    origin="upper",
    transpose=True,
    show_stats=True,
):
    """
    Plot a size x size grid from 3D ndarray data [N, H, W].
    """
    imgs = np.asarray(imgs)

    if imgs.ndim != 3:
        raise ValueError(f"Expected `imgs` to have shape [N,H,W], got {imgs.shape}")

    if size <= 0:
        raise ValueError(f"`size` must be > 0, got {size}")

    num_plots = size * size
    selected = imgs[:num_plots]

    if selected.shape[0] == 0:
        raise ValueError("`imgs` is empty")

    if vmin is None:
        vmin = float(np.min(selected))
    if vmax is None:
        vmax = float(np.max(selected))

    fig, axes = plt.subplots(
        size,
        size,
        figsize=(size * 4, size * 4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for offset, ax in enumerate(axes):
        if offset >= selected.shape[0]:
            ax.axis("off")
            continue

        patch = selected[offset]
        image = patch.T if transpose else patch
        im = ax.imshow(image, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
        if show_stats:
            ax.set_title(
                "\n".join(
                    [
                        f"idx={offset}",
                        f"min={patch.min():.4f}",
                        f"max={patch.max():.4f}",
                        f"mean={patch.mean():.4f}",
                    ]
                ),
                fontsize=9,
            )
        else:
            ax.set_title(f"{title} - {offset}" if title else f"{offset}")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
    if fig_name is None:
        plt.show()
    else:
        fig.savefig(fig_name, dpi=200, bbox_inches="tight")
        print(f"Saved patch grid to: {fig_name}")
    plt.close(fig)
    return fig_name
