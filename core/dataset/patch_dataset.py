import bisect
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Optimized Dataset for loading patches stored in NPY files.
    Designed for large datasets and multi-worker DataLoader.
    """

    def __init__(self, data_path, transform=None, npy=True, verbose=False):

        self.transform = transform
        self.data_path = Path(data_path)
        self.verbose = verbose
        self.npy = bool(npy)
        file_pattern = "*.npy" if self.npy else "*.npz"

        # 文件缓存（每个 worker 独立）
        self.file_cache = {}

        # 获取 patch 文件
        if self.data_path.is_file():
            raise ValueError(
                f"PatchDataset expects a directory, got file: {self.data_path}"
            )
        self.patch_files = sorted(glob(str(self.data_path / file_pattern)))

        if len(self.patch_files) == 0:
            raise ValueError(f"No {file_pattern} patch files found in {data_path}")

        if self.verbose:
            print("Indexing patch files...")

        # 构建索引
        self.file_indices = []
        self.cumulative_sizes = [0]

        for patch_file in self.patch_files:
            # 只读取 metadata
            if self.npy:
                data = np.load(patch_file, mmap_mode='r')
                num_patches = data.shape[0]
            else:
                with np.load(patch_file, mmap_mode='r') as data:
                    if "patches" not in data:
                        raise KeyError(f"{patch_file} does not contain key 'patches'.")
                    num_patches = data['patches'].shape[0]

            start_idx = self.cumulative_sizes[-1]
            end_idx = start_idx + num_patches

            self.file_indices.append({
                "path": patch_file,
                "start_idx": start_idx,
                "end_idx": end_idx
            })

            self.cumulative_sizes.append(end_idx)

        self.total_patches = self.cumulative_sizes[-1]

        if self.verbose:
            print(f"Found {len(self.patch_files)} files")
            print(f"Total patches: {self.total_patches}")

    def __len__(self):
        return self.total_patches

    def _get_file_index(self, idx):
        """
        Binary search to find which file contains idx
        """
        return bisect.bisect_right(self.cumulative_sizes, idx) - 1

    def _load_file(self, file_path):
        """
        Load patch file with caching
        """
        if file_path not in self.file_cache:
            self.file_cache[file_path] = np.load(file_path, mmap_mode='r')
        return self.file_cache[file_path]

    def __getitem__(self, idx):

        if idx < 0 or idx >= self.total_patches:
            raise IndexError("Index out of range")

        # 找到对应文件
        file_idx = self._get_file_index(idx)
        file_info = self.file_indices[file_idx]

        patch_idx = idx - file_info["start_idx"]

        # 读取文件
        data = self._load_file(file_info["path"])

        if self.npy:
            patch = data[patch_idx]
        else:
            patch = data["patches"][patch_idx]

        # 添加 channel
        if patch.ndim == 2:
            patch = patch[np.newaxis, :, :]

        # mmap_mode="r" returns read-only arrays; copy before torch.from_numpy.
        patch = torch.from_numpy(np.array(patch, copy=True)).float()

        if self.transform:
            patch = self.transform(patch)

        return patch
