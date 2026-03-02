import math
import os
from collections import defaultdict

import numpy as np
import segyio
import torch
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def seed_everything(seed: int = 42, deterministic: bool = False):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 控制
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def extract_patches_with_overlap_2d(data, patch_size, overlap):
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


def reconstruct_from_patches(patches, positions, original_shape):
    """
    patches: [N, ph, pw] numpy array
    positions: [(i, j), ...]
    original_shape: (H, W)
    patch_size: (ph, pw)

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


class segy2ds(Dataset):
    """
    SEGY Dataset
    每个样本是一个炮集：
        - shot_data: [n_traces, n_samples]
        - sx_data:    [n_traces, n_samples]  源 X
        - sy_data:    [n_traces, n_samples]  源 Y
        - rx_data:    [n_traces, n_samples]  接收器 X
        - ry_data:    [n_traces, n_samples]  接收器 Y
    """

    def __init__(self, filename):
        super(segy2ds, self).__init__()
        self.filename = filename

        # 打开文件
        self.f = segyio.open(filename, "r", ignore_geometry=True)
        self.f.mmap()

        self.tracecount = self.f.tracecount
        self.nsamples = len(self.f.samples)

        shot_dict = defaultdict(list)

        for i in range(self.tracecount):
            shot = self.f.header[i][segyio.TraceField.FieldRecord]
            shot_dict[shot].append(i)

        self.shot_keys = sorted(shot_dict.keys())
        self.shot_dict = shot_dict
        self.length = len(self.shot_keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        shot_id = self.shot_keys[idx]
        trace_indices = self.shot_dict[shot_id]

        n_traces = len(trace_indices)
        n_samples = self.nsamples

        # 初始化 5 通道 tensor
        out = np.zeros((5, n_traces, n_samples), dtype=np.float32)

        for t_idx, i in enumerate(trace_indices):
            trace = self.f.trace[i].astype(np.float32)
            header = self.f.header[i]

            sx = header.get(segyio.TraceField.SourceX, 0)
            sy = header.get(segyio.TraceField.SourceY, 0)
            rx = header.get(segyio.TraceField.GroupX, 0)
            ry = header.get(segyio.TraceField.GroupY, 0)

            out[0, t_idx, :] = trace
            out[1, t_idx, :] = sx
            out[2, t_idx, :] = sy
            out[3, t_idx, :] = rx
            out[4, t_idx, :] = ry

        # 增加 batch 维
        out = np.expand_dims(out, axis=0)  # [1, 5, n_traces, n_samples]

        return torch.from_numpy(out)


class PatchDataset(Dataset):
    def __init__(self, npz_dir, channel_first=True):
        """
        npz_dir: 存放所有 shot_XXXX.npz 的文件夹
        channel_first: 是否在返回时增加 channel 维 [1, ph, pw]
        """

        self.npz_dir = npz_dir
        self.files = sorted([os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")])

        # 记录所有 patch 的索引 (file_idx, patch_idx)
        self.idx_map = []

        # 预加载每个文件 patch 数量
        self.patches_per_file = []

        for file_idx, f in enumerate(self.files):
            data = np.load(f, allow_pickle=True)
            n_patch = data["patches"].shape[0]
            self.patches_per_file.append(n_patch)
            for p_idx in range(n_patch):
                self.idx_map.append((file_idx, p_idx))

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        file_idx, patch_idx = self.idx_map[idx]
        f = self.files[file_idx]
        data = np.load(f, allow_pickle=True)
        patch = data["patches"][patch_idx]  # shape: [ph, pw]

        # 转为 float32 tensor
        patch = torch.from_numpy(patch.astype(np.float32))

        # 增加 channel 维度 [1, ph, pw]
        patch = patch.unsqueeze(0)

        return patch


# def test_data():
#     dataset = segy2ds("ma2+GathAP.sgy")
#
#     # set a seed for dataload
#     # use transform to convert data to 512x512, [0-1]
#     # select 80% as sample
#     # plot first sample

if __name__ == "__main__":
    pass
    # # 生成 patches
    # dataset = segy2ds("ma2+GathAP.sgy")
    # data = dataset[100].numpy()
    #
    # # 输出存放目录
    # output_dir = "preprocessed_patches"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # patch_size = (64, 64)
    # overlap = (16, 16)
    #
    # # 遍历整个数据集
    # for idx in tqdm.tqdm(range(len(dataset))):
    #     # 取炮集数据
    #     data = dataset[idx].numpy()  # [1,5,n_traces,n_samples]
    #     shot = data[0, 0, :, :]  # 取炮集通道 -> [n_traces, n_samples]
    #
    #     # 切 patch
    #     patches, positions, original_shape = extract_patches_with_overlap_2d(
    #         shot, patch_size=patch_size, overlap=overlap
    #     )
    #
    #     # patches: (N, H, W)
    #     min_val = patches.min(axis=(1, 2), keepdims=True)
    #     max_val = patches.max(axis=(1, 2), keepdims=True)
    #
    #     range_val = max_val - min_val
    #
    #     # 防止除0
    #     range_val[range_val == 0] = 1.0
    #
    #     patches_norm = (patches - min_val) / range_val
    #
    #     # 保存到硬盘
    #     out_file = os.path.join(output_dir, f"shot_{idx:04d}.npz")
    #     np.savez_compressed(
    #         out_file,
    #         patches=patches_norm,
    #         positions=positions,
    #         original_shape=original_shape
    #     )
    #
    # # 读取dataset
    # dataset = PatchDataset("preprocessed_patches")
    #
    # print("patch 数量:", len(dataset))
    #
    # # 取第一个 patch
    # patch = dataset[0]
    # print(patch.shape)  # [1, ph, pw]，例如 [1, 256, 256]
    #
    # # DataLoader
    # from torch.tools.data import DataLoader
    #
    # loader = DataLoader(dataset, batch_size=16, shuffle=True)
    #
    # # 取一个 batch
    # batch = next(iter(loader))  # [B, 1, H, W]
    #
    # num_show = min(16, batch.shape[0])  # 最多16个
    #
    # plt.figure(figsize=(10, 10))
    #
    # for i in range(num_show):
    #     plt.subplot(4, 4, i + 1)
    #
    #     patch = batch[i, 0].cpu().numpy()
    #
    #     plt.imshow(
    #         patch,
    #         cmap='seismic',
    #         aspect='auto',
    #         vmin=0.0,  # 你的数据是0~1
    #         vmax=1.0,
    #         origin='upper'
    #     )
    #
    #     plt.title(f"{i}")
    #     plt.axis("off")
    #
    # plt.tight_layout()
    # plt.show()
