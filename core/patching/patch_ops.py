import math
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch


class BasePatchProcessor(ABC):
    def _compute_patch_grid_2d(self, height, width, patch_size, overlap):
        ph, pw = patch_size
        oh, ow = overlap

        sh = ph - oh
        sw = pw - ow
        if sh <= 0 or sw <= 0:
            raise ValueError("overlap must be smaller than patch_size in both dimensions.")

        n_h = math.ceil((height - ph) / sh) + 1
        n_w = math.ceil((width - pw) / sw) + 1
        return [(ih * sh, iw * sw) for ih in range(n_h) for iw in range(n_w)]

    def _gaussian_2d(self, ph, pw):
        y = np.linspace(-1, 1, ph)
        x = np.linspace(-1, 1, pw)
        xv, yv = np.meshgrid(x, y)
        sigma = 0.5
        gaussian = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.max()
        return gaussian

    @abstractmethod
    def extract_overlapping_patches_2d(self, data, patch_size, overlap):
        raise NotImplementedError

    @abstractmethod
    def reconstruct_from_overlapping_patches_2d(self, patches, positions, original_shape):
        raise NotImplementedError

    def extract_overlapping_patches_3d(self, data, patch_size, overlap):
        raise NotImplementedError("3D patch extraction is not implemented yet.")

    def reconstruct_from_overlapping_patches_3d(
            self, patches, positions, original_shape
    ):
        raise NotImplementedError("3D patch reconstruction is not implemented yet.")


class NumpyPatchProcessor(BasePatchProcessor):
    def extract_overlapping_patches_2d(self, data, patch_size, overlap):
        """
        data: 2D numpy array [H, W]
        patch_size: (ph, pw)
        overlap: (oh, ow)

        return:
            patches: [N, ph, pw]
            positions: [N, 2]
            original_shape: (H, W)
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray.")
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D numpy array [H, W], got ndim={data.ndim}.")

        height, width = data.shape
        ph, pw = patch_size
        positions = self._compute_patch_grid_2d(height, width, patch_size, overlap)

        patches = []
        for i, j in positions:
            patch = np.zeros((ph, pw), dtype=data.dtype)
            i_end = min(i + ph, height)
            j_end = min(j + pw, width)
            h_valid = i_end - i
            w_valid = j_end - j
            patch[0:h_valid, 0:w_valid] = data[i:i_end, j:j_end]
            patches.append(patch)

        return np.stack(patches), np.asarray(positions, dtype=np.int64), (height, width)

    def reconstruct_from_overlapping_patches_2d(self, patches, positions, original_shape):
        """
        patches: [N, ph, pw] numpy array
        positions: [N, 2]
        original_shape: (H, W)

        return:
            reconstructed: [H, W] numpy array
        """

        if not isinstance(patches, np.ndarray):
            raise TypeError("patches must be a numpy.ndarray.")
        if patches.ndim != 3:
            raise ValueError(f"Expected patches with shape [N, ph, pw], got ndim={patches.ndim}.")

        positions_array = np.asarray(positions, dtype=np.int64)
        if positions_array.ndim != 2 or positions_array.shape[1] != 2:
            raise ValueError("positions must have shape [num_patches, 2].")
        if positions_array.shape[0] != patches.shape[0]:
            raise ValueError("positions length must match the number of patches.")

        height, width = original_shape
        _, ph, pw = patches.shape
        weight_patch = self._gaussian_2d(ph, pw).astype(np.float32)

        reconstructed = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        for patch, (i, j) in zip(patches, positions_array):
            i_end = min(i + ph, height)
            j_end = min(j + pw, width)
            h_valid = i_end - i
            w_valid = j_end - j

            reconstructed[i:i_end, j:j_end] += (
                    patch[0:h_valid, 0:w_valid] * weight_patch[0:h_valid, 0:w_valid]
            )
            weight_sum[i:i_end, j:j_end] += weight_patch[0:h_valid, 0:w_valid]

        weight_sum[weight_sum == 0] = 1.0
        return reconstructed / weight_sum


class TensorPatchProcessor(BasePatchProcessor):
    def extract_overlapping_patches_2d(self, data, patch_size, overlap):
        """
        data: 4D torch tensor [B, C, H, W]
        patch_size: (ph, pw)
        overlap: (oh, ow)

        return:
            patches: [B, N, C, ph, pw]
            positions: [N, 2]
            original_shape: (H, W)
        """

        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor.")
        if data.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, C, H, W], got ndim={data.ndim}.")

        batch_size, channels, height, width = data.shape
        ph, pw = patch_size
        positions = self._compute_patch_grid_2d(height, width, patch_size, overlap)

        patches = []
        for i, j in positions:
            patch = torch.zeros(
                (batch_size, channels, ph, pw),
                dtype=data.dtype,
                device=data.device,
            )
            i_end = min(i + ph, height)
            j_end = min(j + pw, width)
            h_valid = i_end - i
            w_valid = j_end - j
            patch[:, :, 0:h_valid, 0:w_valid] = data[:, :, i:i_end, j:j_end]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)
        positions = torch.tensor(positions, dtype=torch.long, device=data.device)
        return patches, positions, (height, width)

    def reconstruct_from_overlapping_patches_2d(self, patches, positions, original_shape):
        """
        patches: [B, N, C, ph, pw] torch tensor
        positions: [N, 2]
        original_shape: (H, W)

        return:
            reconstructed: [B, C, H, W] torch tensor
        """

        if not isinstance(patches, torch.Tensor):
            raise TypeError("patches must be a torch.Tensor.")
        if patches.ndim != 5:
            raise ValueError(
                f"Expected patches with shape [B, N, C, ph, pw], got ndim={patches.ndim}."
            )

        batch_size, num_patches, channels, ph, pw = patches.shape
        height, width = original_shape

        if isinstance(positions, torch.Tensor):
            positions_tensor = positions.to(device=patches.device, dtype=torch.long)
        else:
            positions_tensor = torch.tensor(positions, dtype=torch.long, device=patches.device)

        if positions_tensor.ndim != 2 or positions_tensor.shape[1] != 2:
            raise ValueError("positions must have shape [num_patches, 2].")
        if positions_tensor.shape[0] != num_patches:
            raise ValueError("positions length must match the number of patches.")

        weight_patch = torch.from_numpy(self._gaussian_2d(ph, pw)).to(
            device=patches.device,
            dtype=patches.dtype,
        )
        weight_patch = weight_patch.unsqueeze(0).unsqueeze(0)

        reconstructed = torch.zeros(
            (batch_size, channels, height, width),
            dtype=patches.dtype,
            device=patches.device,
        )
        weight_sum = torch.zeros(
            (batch_size, 1, height, width),
            dtype=patches.dtype,
            device=patches.device,
        )

        for patch_idx in range(num_patches):
            i, j = positions_tensor[patch_idx].tolist()
            i_end = min(i + ph, height)
            j_end = min(j + pw, width)
            h_valid = i_end - i
            w_valid = j_end - j

            patch = patches[:, patch_idx, :, 0:h_valid, 0:w_valid]
            weight = weight_patch[:, :, 0:h_valid, 0:w_valid]
            reconstructed[:, :, i:i_end, j:j_end] += patch * weight
            weight_sum[:, :, i:i_end, j:j_end] += weight

        weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
        return reconstructed / weight_sum


NUMPY_PATCH_PROCESSOR = NumpyPatchProcessor()
TENSOR_PATCH_PROCESSOR = TensorPatchProcessor()


def extract_overlapping_patches_2d(data, patch_size, overlap):
    warnings.warn(
        "extract_overlapping_patches_2d is deprecated and will be removed in a future version. "
        "Use NumpyPatchProcessor.extract_overlapping_patches_2d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return NUMPY_PATCH_PROCESSOR.extract_overlapping_patches_2d(data, patch_size, overlap)


def reconstruct_from_overlapping_patches_2d(patches, positions, original_shape):
    warnings.warn(
        "reconstruct_from_overlapping_patches_2d is deprecated and will be removed in a future version. "
        "Use NumpyPatchProcessor.reconstruct_from_overlapping_patches_2d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return NUMPY_PATCH_PROCESSOR.reconstruct_from_overlapping_patches_2d(
        patches, positions, original_shape
    )


def extract_overlapping_patches_2d_tensor(data, patch_size, overlap):
    warnings.warn(
        "extract_overlapping_patches_2d_tensor is deprecated and will be removed in a future version. "
        "Use TensorPatchProcessor.extract_overlapping_patches_2d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TENSOR_PATCH_PROCESSOR.extract_overlapping_patches_2d(data, patch_size, overlap)


def reconstruct_from_overlapping_patches_2d_tensor(patches, positions, original_shape):
    warnings.warn(
        "reconstruct_from_overlapping_patches_2d_tensor is deprecated and will be removed in a future version. "
        "Use TensorPatchProcessor.reconstruct_from_overlapping_patches_2d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TENSOR_PATCH_PROCESSOR.reconstruct_from_overlapping_patches_2d(
        patches, positions, original_shape
    )


if __name__ == "__main__":
    from matplotlib import pyplot as plt


    def test_numpy_patch_processor():
        processor = NumpyPatchProcessor()

        raw = np.zeros((256, 256), dtype=np.float32)
        for i in range(256):
            for j in range(256):
                raw[i, j] = i + 2 * j

        vmax = raw.max()
        vmin = raw.min()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(raw, cmap="seismic", aspect=1, vmin=vmin, vmax=vmax)
        plt.title("raw numpy")
        plt.tick_params(axis="both", labelsize=18)
        plt.show()

        patches, positions, original_shape = processor.extract_overlapping_patches_2d(
            raw,
            (64, 64),
            (16, 16),
        )
        plt.figure(figsize=(8, 8))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(patches[i], cmap="seismic", aspect=1, vmax=vmax, vmin=vmin)
            plt.title(f"{i}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        recon = processor.reconstruct_from_overlapping_patches_2d(
            patches,
            positions,
            original_shape,
        )
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(recon, cmap="seismic", aspect=1, vmin=vmin, vmax=vmax)
        plt.title("recon numpy")
        plt.tick_params(axis="both", labelsize=18)
        plt.show()

        diff = raw - recon
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(diff, cmap="seismic", aspect=1, vmin=vmin, vmax=vmax)
        plt.title("diff numpy")
        plt.tick_params(axis="both", labelsize=18)
        plt.show()

        print(f"[numpy] reconstruct residual: max={diff.max()}, min={diff.min()}")


    def test_tensor_patch_processor():
        processor = TensorPatchProcessor()

        raw = np.zeros((256, 256), dtype=np.float32)
        for i in range(256):
            for j in range(256):
                raw[i, j] = i + 2 * j

        tensor_data = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)
        vmax = float(tensor_data.max())
        vmin = float(tensor_data.min())

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(tensor_data[0, 0].numpy(), cmap="seismic", aspect=1, vmin=vmin, vmax=vmax)
        plt.title("raw tensor")
        plt.tick_params(axis="both", labelsize=18)
        plt.show()

        patches, positions, original_shape = processor.extract_overlapping_patches_2d(
            tensor_data,
            (64, 64),
            (16, 16),
        )
        plt.figure(figsize=(8, 8))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(
                patches[0, i, 0].numpy(),
                cmap="seismic",
                aspect=1,
                vmax=vmax,
                vmin=vmin,
            )
            plt.title(f"{i}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        recon = processor.reconstruct_from_overlapping_patches_2d(
            patches,
            positions,
            original_shape,
        )
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(recon[0, 0].numpy(), cmap="seismic", aspect=1, vmin=vmin, vmax=vmax)
        plt.title("recon tensor")
        plt.tick_params(axis="both", labelsize=18)
        plt.show()

        diff = tensor_data - recon
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(
            diff[0, 0].numpy(),
            cmap="seismic",
            aspect=1,
            vmin=vmin,
            vmax=vmax,
        )
        plt.title("diff tensor")
        plt.tick_params(axis="both", labelsize=18)
        plt.show()

        print(
            f"[tensor] reconstruct residual: max={float(diff.max())}, min={float(diff.min())}"
        )


    test_numpy_patch_processor()
    test_tensor_patch_processor()
