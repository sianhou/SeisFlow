from collections import defaultdict

import numpy as np
import segyio
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

from core.seed_everything import seed_everything


class MinMaxToMinus1Plus1:
    def __init__(self, eps=1e-12):
        self.eps = eps

    def __call__(self, x):
        if x.ndim == 3:  # [C,H,W]
            c0 = x[0]
            mn = c0.amin()
            mx = c0.amax()
            x0 = 2 * (c0 - mn) / (mx - mn + self.eps) - 1
            x = x.clone()
            x[0] = x0
            return x

        if x.ndim == 4:  # [B,C,H,W]
            c0 = x[:, 0]  # [B,H,W]
            mn = c0.amin(dim=(1, 2), keepdim=True)
            mx = c0.amax(dim=(1, 2), keepdim=True)
            x0 = 2 * (c0 - mn) / (mx - mn + self.eps) - 1
            x = x.clone()
            x[:, 0] = x0
            return x


class ClipFirstChannel:

    def __init__(self, vmin=None, vmax=None, inplace=False):
        assert vmin is not None or vmax is not None, "vmin/vmax 至少提供一个"
        self.vmin = vmin
        self.vmax = vmax
        self.inplace = inplace

    def __call__(self, x: torch.Tensor):
        if not self.inplace:
            x = x.clone()

        if x.ndim == 3:  # [C,H,W]
            x[0] = x[0].clamp(self.vmin, self.vmax)
        elif x.ndim == 4:  # [B,C,H,W] 或 [1,C,H,W]
            x[:, 0] = x[:, 0].clamp(self.vmin, self.vmax)
        elif x.ndim == 5:  # [B,1,C,H,W]
            x[:, :, 0] = x[:, :, 0].clamp(self.vmin, self.vmax)
        else:
            raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

        return x


class ScaleFirstChannel:
    def __init__(self, scalar=1.0):
        self.scalar = scalar

    def __call__(self, x: torch.Tensor):
        x = x.clone()

        if x.ndim == 3:  # [C,H,W]
            x[0] = x[0] * self.scalar
        elif x.ndim == 4:  # [B,C,H,W] 或 [1,C,H,W]
            x[:, 0] = x[:, 0] * self.scalar
        elif x.ndim == 5:  # [B,1,C,H,W]
            x[:, :, 0] = x[:, :, 0] * self.scalar
        else:
            raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

        return x


class SliceLastDim:
    """
    只截取最后一维：x[..., start:end]
    - start/end 支持负数
    - end=None 表示到结尾
    """

    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def __call__(self, x: torch.Tensor):
        return x[..., self.start:self.end]


class SegyDataset(Dataset):
    def __init__(self, filename, transform=None, add_batch_dim=False):
        super(SegyDataset, self).__init__()
        self.filename = filename
        self.transform = transform
        self.add_batch_dim = add_batch_dim

        shot_dict = defaultdict(list)
        with segyio.open(filename, "r", ignore_geometry=True) as f:
            tracecount = f.tracecount
            nsamples = len(f.samples)

            for i in range(tracecount):
                shot = f.header[i][segyio.TraceField.FieldRecord]
                shot_dict[shot].append(i)

        self.shot_keys = sorted(shot_dict.keys())
        self.shot_dict = shot_dict
        self.length = len(self.shot_keys)
        self.tracecount = tracecount
        self.nsamples = nsamples

        # worker 内懒加载用
        self.f = None

    def __len__(self):
        return self.length

    def _ensure_open(self):
        if self.f is None:
            self.f = segyio.open(self.filename, "r", ignore_geometry=True)
            self.f.mmap()

    def __getitem__(self, idx):
        self._ensure_open()

        shot_id = self.shot_keys[idx]
        trace_indices = self.shot_dict[shot_id]

        n_traces = len(trace_indices)
        n_samples = self.nsamples

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

        x = torch.from_numpy(out)  # [5, n_traces, n_samples]

        if self.add_batch_dim:
            x = x.unsqueeze(0)  # [1, 5, n_traces, n_samples]

        if self.transform is not None:
            x = self.transform(x)

        return x


if __name__ == "__main__":
    seed_everything(0)

    transform = transforms.Compose([
        SliceLastDim(0, 1501),
        ClipFirstChannel(-2, 2),
        transforms.Resize((512, 512)),
    ])

    dataset = SegyDataset("../ma2+GathAP.sgy", transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    x = next(iter(dataloader))

    imgs = x[:, 0]  # [B,H,W]
    imgs = imgs.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    for i in range(4):
        axes[i].imshow(imgs[i].T, cmap="seismic", origin="upper", vmin=-1, vmax=1)
        axes[i].set_title(f"sample {i} (ch0)")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
