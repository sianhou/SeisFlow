import torch
from diffusers.models.activations import get_activation
from torch import nn


class FactorizedConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int, int] = 3,
            stride: int | tuple[int, int, int] = 1,
            padding: int | tuple[int, int, int] = 1,
            dilation: int | tuple[int, int, int] = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        super().__init__()

        kernel_size = self._triple(kernel_size)
        stride = self._triple(stride)
        padding = self._triple(padding)
        dilation = self._triple(dilation)

        if any(size <= 0 for size in kernel_size):
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")

        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            dilation=(dilation[0], 1, 1),
            groups=groups,
            bias=False,
        )
        self.spatial_conv = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            dilation=(1, dilation[1], dilation[2]),
            groups=groups,
            bias=bias,
        )

    @staticmethod
    def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError(f"Expected a 3-tuple, got {value}.")
            return value
        return value, value, value

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.depth_conv(input_tensor)
        hidden_states = self.spatial_conv(hidden_states)
        return hidden_states


def _make_conv3d(
        conv_func: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
) -> nn.Module:
    if conv_func == "conv":
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    if conv_func == "factorized":
        return FactorizedConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    raise ValueError(f"Unsupported conv_func: {conv_func}. Expected 'conv' or 'factorized'.")


class ResnetBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int | None = None,
            dropout: float = 0.0,
            groups: int = 32,
            groups_out: int | None = None,
            eps: float = 1e-6,
            non_linearity: str = "swish",
            conv_func: str = "conv",
            conv_kernel_size: int = 3,
            output_scale_factor: float = 1.0,
            conv_shortcut_bias: bool = True,
            conv_3d_out_channels: int | None = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        groups_out = groups if groups_out is None else groups_out
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be a positive odd integer, got {conv_kernel_size}.")
        conv_padding = conv_kernel_size // 2
        self.conv_shortcut = None

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = _make_conv3d(
            conv_func=conv_func,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
        )
        self.norm2 = nn.GroupNorm(num_groups=min(groups_out, out_channels), num_channels=out_channels, eps=eps,
                                  affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = _make_conv3d(
            conv_func=conv_func,
            in_channels=out_channels,
            out_channels=conv_3d_out_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
        )
        self.nonlinearity = get_activation(non_linearity)

        if in_channels != conv_3d_out_channels:
            self.conv_shortcut = nn.Conv3d(in_channels=in_channels, out_channels=conv_3d_out_channels, kernel_size=1,
                                           stride=1, padding=0, bias=conv_shortcut_bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            if self.training:
                input_tensor = input_tensor.contiguous()
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


def test_ResnetBlock3D():
    in_channels = 8
    out_channels = 16
    groups = 4
    dropout = 0.0
    eps = 1e-6

    x = torch.randn(2, in_channels, 8, 32, 32)
    expected_shape = (2, out_channels, 8, 32, 32)

    for conv_func in ("conv", "factorized"):
        for conv_kernel_size in (3, 5, 7):
            torch.manual_seed(0)
            resnet_block = ResnetBlock3D(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                groups_out=groups,
                dropout=dropout,
                eps=eps,
                non_linearity="swish",
                conv_func=conv_func,
                conv_kernel_size=conv_kernel_size,
                output_scale_factor=1.0,
            )

            resnet_block.eval()

            with torch.no_grad():
                y = resnet_block(x)

            print(f"{conv_func=}, {conv_kernel_size=}")
            print("input shape:", x.shape)
            print("output shape:", y.shape)
            print("all finite:", torch.isfinite(y).all().item())

            assert y.shape == expected_shape
            assert torch.isfinite(y).all()

            if conv_func == "factorized":
                assert isinstance(resnet_block.conv1, FactorizedConv3d)
                assert isinstance(resnet_block.conv2, FactorizedConv3d)


if __name__ == "__main__":
    test_ResnetBlock3D()
