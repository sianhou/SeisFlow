class MinMaxToMinusOneOne:
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
