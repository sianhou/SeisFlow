import torch


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
