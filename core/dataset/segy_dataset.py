from collections import defaultdict

import numpy as np
import segyio
import torch
from torch.utils.data import Dataset


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

    from matplotlib import pyplot as plt
    from torchvision import transforms

    from core.training.seed import set_random_seed
    from core.transforms import ClipFirstChannel, SliceLastDimension

    set_random_seed(0)

    transform = transforms.Compose([
        SliceLastDimension(0, 1501),
        ClipFirstChannel(-2, 2),
        transforms.Resize((512, 512)),
    ])

    dataset = SegyDataset("../../ma2+GathAP.sgy", transform=transform)

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
