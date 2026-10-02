"""Plot 16 PatchDataset samples in a 4 × 4 grid."""

import matplotlib.pyplot as plt

from core.dataset import PatchDataset


def plot_patches(dataset, indices=None, vmin=-1, vmax=1, title=""):
    """Plot seismic patches with thin borders and a shared amplitude scale.

    Args:
        dataset: PatchDataset returning tensors shaped (1, traces, samples).
        indices: Sequence of 16 zero-based indices in row-major order. None
            selects the first 16 samples.
        vmin: Lower amplitude bound of the shared color scale.
        vmax: Upper amplitude bound of the shared color scale.
        title: Overall figure title. An empty string leaves the title hidden.

    Returns:
        Matplotlib figure containing an unlabeled 4 × 4 grid with narrow gaps,
        thin borders, a shared colorbar, and time pointing downward.
    """
    if indices is None:
        indices = range(16)
    patches = [dataset[index][0].numpy().T for index in indices]
    figure, axes = plt.subplots(4, 4, figsize=(13, 12))
    figure.subplots_adjust(
        left=0.005, right=0.92, bottom=0.005, top=0.995,
        wspace=0.02, hspace=0.02,
    )
    if title:
        figure.suptitle(title)
        figure.subplots_adjust(top=0.95)

    for axis, patch in zip(axes.flat, patches):
        image = axis.imshow(
            patch,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)

    colorbar_axis = figure.add_axes([0.94, 0.06, 0.015, 0.88])
    figure.colorbar(image, cax=colorbar_axis)
    return figure


if __name__ == "__main__":
    input_data_dir = (
        "/Users/housian/Workplaces/SeisFlow/temp/"
        "shot_dataset128_overlap64_31shots/train"
    )
    dataset = PatchDataset(input_data_dir)
    print(len(dataset))
    figure = plot_patches(dataset)
    plt.show()
