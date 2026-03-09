import torch


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
