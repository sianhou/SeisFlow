import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.patching import NumpyPatchProcessor, TensorPatchProcessor


def build_sin_target(height=73, width=91):
    y = np.linspace(0.0, 2.0 * np.pi, height, dtype=np.float32)
    x = np.linspace(0.0, 4.0 * np.pi, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (
            np.sin(xx)
            + 0.5 * np.cos(yy)
            + 0.25 * np.sin(xx + yy)
    ).astype(np.float32)


def test_numpy_patch_processor_reconstructs_sin_target():
    target = build_sin_target()
    processor = NumpyPatchProcessor()

    patches, positions, original_shape = processor.extract_overlapping_patches_2d(
        target,
        patch_size=(32, 32),
        overlap=(8, 8),
    )
    reconstructed = processor.reconstruct_from_overlapping_patches_2d(
        patches,
        positions,
        original_shape,
    )

    assert patches.shape[1:] == (32, 32)
    assert original_shape == target.shape
    assert positions.shape[1] == 2
    assert np.allclose(reconstructed, target, atol=1e-6)


def test_tensor_patch_processor_reconstructs_sin_target():
    target = build_sin_target()
    tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    processor = TensorPatchProcessor()

    patches, positions, original_shape = processor.extract_overlapping_patches_2d(
        tensor,
        patch_size=(32, 32),
        overlap=(8, 8),
    )
    reconstructed = processor.reconstruct_from_overlapping_patches_2d(
        patches,
        positions,
        original_shape,
    )

    assert patches.shape[2:] == (1, 32, 32)
    assert original_shape == target.shape
    assert positions.shape[1] == 2
    assert torch.allclose(reconstructed, tensor, atol=1e-6)


def _plot_reconstruction(target, numpy_recon, tensor_recon, output_path):
    import matplotlib.pyplot as plt

    numpy_diff = numpy_recon - target
    tensor_diff = tensor_recon - target
    vlim = max(float(np.abs(target).max()), 1e-6)
    diff_vlim = max(
        float(np.abs(numpy_diff).max()),
        float(np.abs(tensor_diff).max()),
        1e-8,
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        (target, "Target", -vlim, vlim),
        (numpy_recon, "Numpy recon", -vlim, vlim),
        (numpy_diff, "Numpy diff", -diff_vlim, diff_vlim),
        (target, "Target", -vlim, vlim),
        (tensor_recon, "Tensor recon", -vlim, vlim),
        (tensor_diff, "Tensor diff", -diff_vlim, diff_vlim),
    ]

    for ax, (image, title, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_patch_processor_demo(
        height=73,
        width=91,
        patch_size=(32, 32),
        overlap=(8, 8),
        output_dir="./",
):
    target = build_sin_target(height=height, width=width)

    numpy_processor = NumpyPatchProcessor()
    numpy_patches, numpy_positions, numpy_shape = (
        numpy_processor.extract_overlapping_patches_2d(
            target,
            patch_size=patch_size,
            overlap=overlap,
        )
    )
    numpy_recon = numpy_processor.reconstruct_from_overlapping_patches_2d(
        numpy_patches,
        numpy_positions,
        numpy_shape,
    )

    tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    tensor_processor = TensorPatchProcessor()
    tensor_patches, tensor_positions, tensor_shape = (
        tensor_processor.extract_overlapping_patches_2d(
            tensor,
            patch_size=patch_size,
            overlap=overlap,
        )
    )
    tensor_recon = tensor_processor.reconstruct_from_overlapping_patches_2d(
        tensor_patches,
        tensor_positions,
        tensor_shape,
    ).squeeze(0).squeeze(0).detach().cpu().numpy()

    numpy_max_abs_error = float(np.abs(numpy_recon - target).max())
    tensor_max_abs_error = float(np.abs(tensor_recon - target).max())

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="patch_processor_demo_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "patch_processor_reconstruction.png"
    _plot_reconstruction(target, numpy_recon, tensor_recon, output_path)

    return {
        "target_shape": target.shape,
        "patch_size": patch_size,
        "overlap": overlap,
        "numpy_num_patches": int(numpy_patches.shape[0]),
        "tensor_num_patches": int(tensor_patches.shape[1]),
        "numpy_max_abs_error": numpy_max_abs_error,
        "tensor_max_abs_error": tensor_max_abs_error,
        "figure_path": output_path,
    }


if __name__ == "__main__":
    result = run_patch_processor_demo()
    print("PatchProcessor reconstruction demo")
    print(f"  target_shape: {result['target_shape']}")
    print(f"  patch_size: {result['patch_size']}")
    print(f"  overlap: {result['overlap']}")
    print(f"  numpy_num_patches: {result['numpy_num_patches']}")
    print(f"  tensor_num_patches: {result['tensor_num_patches']}")
    print(f"  numpy_max_abs_error: {result['numpy_max_abs_error']:.8e}")
    print(f"  tensor_max_abs_error: {result['tensor_max_abs_error']:.8e}")
    print(f"  figure_path: {result['figure_path']}")
