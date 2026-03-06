from matplotlib import pyplot as plt


def plot2x2(imgs, fig_name, title="", vmin=-1, vmax=1, ):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()

    for i in range(4):
        axes[i].imshow(imgs[i].T, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{title} - {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(fig_name)


def plot_one_row(imgs, fig_name, title="", vmin=-1, vmax=1, ):
    num_figs = imgs.shape[0]

    fig, axes = plt.subplots(1, num_figs, figsize=(6, 12))
    axes = axes.ravel()

    for i in range(num_figs):
        axes[i].imshow(imgs[i].T, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(fig_name)
