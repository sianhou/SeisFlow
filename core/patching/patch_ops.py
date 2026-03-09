import math

import numpy as np


def extract_overlapping_patches_2d(data, patch_size, overlap):
    """
    data: 2D numpy array [H, W]
    patch_size: (ph, pw)
    overlap: (oh, ow)

    return:
        patches: [N, ph, pw]
        positions: [(i, j)]
    """

    H, W = data.shape
    ph, pw = patch_size
    oh, ow = overlap

    sh = ph - oh
    sw = pw - ow

    n_h = math.ceil((H - ph) / sh) + 1
    n_w = math.ceil((W - pw) / sw) + 1

    patches = []
    positions = []

    for ih in range(n_h):
        for iw in range(n_w):
            i = ih * sh
            j = iw * sw

            patch = np.zeros((ph, pw), dtype=data.dtype)

            i_end = min(i + ph, H)
            j_end = min(j + pw, W)

            h_valid = i_end - i
            w_valid = j_end - j

            patch[0:h_valid, 0:w_valid] = data[i:i_end, j:j_end]

            patches.append(patch)
            positions.append((i, j))

    patches = np.stack(patches)

    original_shape = (H, W)

    return patches, positions, original_shape


def reconstruct_from_overlapping_patches_2d(patches, positions, original_shape):
    """
    patches: [N, ph, pw] numpy array
    positions: [(i, j), ...]
    original_shape: (H, W)

    return:
        reconstructed: [H, W] numpy array
    """

    H, W = original_shape
    ph, pw = patches[0].shape

    # 构建高斯权重窗
    def gaussian_2d(ph, pw):
        y = np.linspace(-1, 1, ph)
        x = np.linspace(-1, 1, pw)
        xv, yv = np.meshgrid(x, y)
        sigma = 0.5
        g = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        g = g / g.max()
        return g

    weight_patch = gaussian_2d(ph, pw)

    # 初始化输出矩阵和权重矩阵
    reconstructed = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for p, (i, j) in zip(patches, positions):
        i_end = min(i + ph, H)
        j_end = min(j + pw, W)

        h_valid = i_end - i
        w_valid = j_end - j

        reconstructed[i:i_end, j:j_end] += p[0:h_valid, 0:w_valid] * weight_patch[0:h_valid, 0:w_valid]
        weight_sum[i:i_end, j:j_end] += weight_patch[0:h_valid, 0:w_valid]

    # 避免除零
    weight_sum[weight_sum == 0] = 1.0
    reconstructed /= weight_sum

    return reconstructed


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    # generate test data
    raw = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            raw[i, j] = i + 2 * j
    vmax = raw.max()
    vmin = raw.min()

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(raw, cmap='seismic', aspect=1, vmin=vmin, vmax=vmax)
    plt.title(f"raw")
    plt.tick_params(axis='both', labelsize=18)
    plt.show()

    # generate patches
    patches, positions, original_shape = extract_overlapping_patches_2d(raw, (64, 64), (16, 16))
    plt.figure(figsize=(8, 8))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        patch = patches[i]
        plt.imshow(patch, cmap='seismic', aspect=1, vmax=vmax, vmin=vmin, )
        plt.title(f"{i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # reconstruct
    recon = reconstruct_from_overlapping_patches_2d(patches, positions, original_shape)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(recon, cmap='seismic', aspect=1, vmin=vmin, vmax=vmax)
    plt.title(f"recon")
    plt.tick_params(axis='both', labelsize=18)
    plt.show()

    # difference
    diff = raw - recon

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(diff, cmap='seismic', aspect=1, vmin=vmin, vmax=vmax)
    plt.title(f"diff")
    plt.tick_params(axis='both', labelsize=18)
    plt.show()

    vmax = diff.max()
    vmin = diff.min()

    print(f"reconstruct residual: max={vmax}, min={vmin}")
