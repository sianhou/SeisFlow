import torch


class AbsNormalize:
    """
    Absolute-value normalization for torch tensors.

    Expected input shape: ``[B, C, ...]`` with ``ndim >= 3``. The first
    dimension is always treated as batch, and the second dimension is always
    treated as channel.

    The transform divides the input by an absolute scale computed from the
    input itself. ``run`` returns both the normalized tensor and the broadcastable
    scale, so ``normalized * scale`` reconstructs the original values.

    Example:
        >>> x = torch.randn(2, 3, 64, 64)
        >>> transform = AbsNormalize(per_channel=True)
        >>> normalized = transform(x)
        >>> normalized, scale = transform.run(x)
        >>> restored = normalized * scale
    """

    def __init__(self, per_channel=True, eps=1e-12):
        self.per_channel = per_channel
        self.eps = eps

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 3:
            raise ValueError("Input tensor must have shape [B, C, ...] with ndim >= 3")

    def _batch_dim(self, x):
        return 0

    def _channel_dim(self, x):
        return 1

    def _reduce_dims(self, x):
        batch_dim = self._batch_dim(x)

        if not self.per_channel:
            return tuple(dim for dim in range(x.ndim) if dim != batch_dim)

        channel_dim = self._channel_dim(x)
        return tuple(
            dim for dim in range(x.ndim) if dim not in {batch_dim, channel_dim}
        )

    def get_scale(self, x):
        self._validate_input(x)

        reduce_dims = self._reduce_dims(x)
        scale = x.abs().amax(dim=reduce_dims, keepdim=True)
        return scale.clamp_min(self.eps)

    def __call__(self, x):
        normalized, _ = self.run(x)
        return normalized

    def run(self, x):
        scale = self.get_scale(x)
        return x / scale, scale


class Normalize:
    """
    Callable transform that normalizes a tensor to [-1, 1].

    Expected input shape: [B, C, ...], ndim >= 4.
    The "abs" method keeps 0 as the symmetry center by dividing by the
    maximum absolute value on the selected reduction axes.
    """

    def __init__(self, mode="per_channel", method="abs", eps=1e-12):
        self.mode = mode.lower()
        self.method = method.lower()
        self.eps = eps

        if self.mode not in {"first_channel", "per_channel", "all_channel", }:
            raise ValueError("mode must be one of: 'first_channel', 'per_channel', 'all_channel'")
        
        if self.method not in {"minmax", "abs", "rms"}:
            raise ValueError("method must be one of: 'minmax', 'abs', 'rms'")

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 4:
            raise ValueError("Input tensor must have shape [B, C, ...] with ndim >= 4")

        spatial_dims = tuple(range(2, x.ndim))

        if self.mode == "first_channel":
            target = x[:, 0:1, ...]
            reduce_dims = spatial_dims
        elif self.mode == "per_channel":
            target = x
            reduce_dims = spatial_dims
        else:
            target = x
            reduce_dims = tuple(range(1, x.ndim))

        if self.method == "minmax":
            mins = target.amin(dim=reduce_dims, keepdim=True)
            maxs = target.amax(dim=reduce_dims, keepdim=True)
            normalized_target = 2.0 * (target - mins) / (maxs - mins + self.eps) - 1.0
        elif self.method == "abs":
            scale = target.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
            normalized_target = (target / scale).clamp(-1.0, 1.0)
        else:
            rms = torch.sqrt(target.pow(2).mean(dim=reduce_dims, keepdim=True)).clamp_min(
                self.eps
            )
            normalized_target = (target / rms).clamp(-1.0, 1.0)

        if self.mode == "first_channel":
            result = x.clone()
            result[:, 0:1, ...] = normalized_target
            return result

        return normalized_target


class MinMaxToMinusOneOne:
    """
    Legacy transform kept for compatibility.

    This class still performs min-max normalization on the first channel.
    Use Normalize(method="abs") when you want zero-centered symmetric scaling.
    """

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


class PerChannelMinMaxToMinusOneOne:
    """
    Legacy transform kept for compatibility.

    This class still performs per-channel min-max normalization.
    Use Normalize(mode="per_channel", method="abs") for zero-centered scaling.
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

    def __call__(self, x):

        # [C,H,W]
        if x.ndim == 3:
            mn = x.amin(dim=(1, 2), keepdim=True)
            mx = x.amax(dim=(1, 2), keepdim=True)

            x = 2 * (x - mn) / (mx - mn + self.eps) - 1
            return x

        # [B,C,H,W]
        elif x.ndim == 4:
            mn = x.amin(dim=(2, 3), keepdim=True)
            mx = x.amax(dim=(2, 3), keepdim=True)

            x = 2 * (x - mn) / (mx - mn + self.eps) - 1
            return x

        else:
            raise ValueError("Input tensor must be [C,H,W] or [B,C,H,W]")
