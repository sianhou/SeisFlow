import torch
from diffusers.models.activations import get_activation
from torch import nn


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
        self.conv_shortcut = None

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(groups_out, out_channels), num_channels=out_channels, eps=eps,
                                  affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=1)
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

    torch.manual_seed(0)
    resnet_block = ResnetBlock3D(
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        groups_out=groups,
        dropout=dropout,
        eps=eps,
        non_linearity="swish",
        output_scale_factor=1.0,
    )

    resnet_block.eval()

    x = torch.randn(2, in_channels, 8, 32, 32)

    with torch.no_grad():
        y = resnet_block(x)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("all finite:", torch.isfinite(y).all().item())


if __name__ == "__main__":
    test_ResnetBlock3D()
