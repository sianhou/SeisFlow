import torch


class Clip:
    """
    Clip torch tensors with shape ``[B, C, ...]`` and ``ndim >= 3``.

    If ``per_channel`` is True, sequence or tensor ``vmin``/``vmax`` values are
    interpreted as channel-wise bounds with length ``C``. Scalar bounds are
    still allowed and are applied to every channel. If ``per_channel`` is False,
    bounds are applied to all channels together.

    Example:
        >>> x = torch.randn(2, 3, 64, 64)
        >>> clipped = Clip(vmin=-2.0, vmax=2.0, per_channel=False)(x)
        >>> clipped = Clip(
        ...     vmin=[-1.0, -2.0, -3.0],
        ...     vmax=[1.0, 2.0, 3.0],
        ...     per_channel=True,
        ... )(x)
    """

    def __init__(self, vmin=None, vmax=None, per_channel=True, inplace=False):
        if vmin is None and vmax is None:
            raise ValueError("vmin/vmax must provide at least one bound")
        self.vmin = vmin
        self.vmax = vmax
        self.per_channel = per_channel
        self.inplace = inplace

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 3:
            raise ValueError("Input tensor must have shape [B, C, ...] with ndim >= 3")

    def _bound_tensor(self, bound, x):
        if bound is None:
            return None

        bound_tensor = torch.as_tensor(bound, dtype=x.dtype, device=x.device)
        if bound_tensor.ndim == 0:
            return bound_tensor

        if not self.per_channel:
            raise ValueError("Sequence bounds require per_channel=True")
        if bound_tensor.numel() != x.shape[1]:
            raise ValueError(
                f"Channel-wise bound length must match C={x.shape[1]}, "
                f"got {bound_tensor.numel()}"
            )

        shape = [1] * x.ndim
        shape[1] = x.shape[1]
        return bound_tensor.reshape(shape)

    def __call__(self, x):
        self._validate_input(x)

        if not self.inplace:
            x = x.clone()

        vmin = self._bound_tensor(self.vmin, x)
        vmax = self._bound_tensor(self.vmax, x)
        return torch.clamp(x, min=vmin, max=vmax)


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
