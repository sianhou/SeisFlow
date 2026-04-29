import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _num_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass
class SeismicVAEConfig:
    input_channels: int = 1
    output_channels: int = 1
    latent_channels: int = 4
    input_size: int = 256
    latent_size: int = 32
    hidden_channels: int = 32
    channel_multipliers: tuple[int, ...] | None = None
    use_vae: bool = True


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            ResBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SeismicSpatialVAE(nn.Module):
    """
    Spatial autoencoder / weak-KL VAE for seismic patches.

    The latent is kept as [B, latent_channels, latent_size, latent_size], so it can
    be consumed by later latent-space flow matching or diffusion modules.
    """

    def __init__(
            self,
            input_channels=1,
            output_channels=1,
            latent_channels=4,
            input_size=256,
            latent_size=32,
            hidden_channels=32,
            channel_multipliers=None,
            use_vae=True,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError("Work Package 1 expects input_channels=1.")
        if input_size <= 0 or latent_size <= 0:
            raise ValueError("input_size and latent_size must be positive.")
        if latent_size not in {16, 32, 64}:
            raise ValueError("latent_size must be one of: 16, 32, 64.")
        if input_size % latent_size != 0:
            raise ValueError("input_size must be divisible by latent_size.")

        compression = input_size // latent_size
        if compression < 1 or compression & (compression - 1) != 0:
            raise ValueError("input_size / latent_size must be a power of two.")

        num_down_blocks = int(math.log2(compression))
        if channel_multipliers is None:
            channel_multipliers = tuple(min(2 ** idx, 8) for idx in range(num_down_blocks + 1))
        if len(channel_multipliers) != num_down_blocks + 1:
            raise ValueError(
                "channel_multipliers length must equal log2(input_size / latent_size) + 1."
            )

        self.config = SeismicVAEConfig(
            input_channels=input_channels,
            output_channels=output_channels,
            latent_channels=latent_channels,
            input_size=input_size,
            latent_size=latent_size,
            hidden_channels=hidden_channels,
            channel_multipliers=tuple(channel_multipliers),
            use_vae=use_vae,
        )
        self.use_vae = use_vae

        encoder_layers = []
        in_ch = input_channels
        for idx, multiplier in enumerate(channel_multipliers):
            out_ch = hidden_channels * multiplier
            stride = 1 if idx == 0 else 2
            encoder_layers.append(ResBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_pool = nn.AdaptiveAvgPool2d((latent_size, latent_size))
        self.to_mu = nn.Conv2d(in_ch, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(in_ch, latent_channels, kernel_size=1)

        decoder_layers = [nn.Conv2d(latent_channels, in_ch, kernel_size=3, padding=1), nn.SiLU()]
        reversed_multipliers = list(reversed(channel_multipliers[:-1]))
        current_channels = in_ch
        for multiplier in reversed_multipliers:
            out_ch = hidden_channels * multiplier
            decoder_layers.append(UpBlock(current_channels, out_ch))
            current_channels = out_ch
        decoder_layers.extend(
            [
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(current_channels, output_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    @classmethod
    def from_config(cls, config: SeismicVAEConfig | dict):
        if isinstance(config, SeismicVAEConfig):
            config = config.__dict__
        return cls(**config)

    def encode(self, x):
        h = self.encoder(x)
        h = self.latent_pool(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(min=-30.0, max=20.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.use_vae or not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon = self.decoder(z)
        if recon.shape[-2:] != (self.config.input_size, self.config.input_size):
            recon = F.interpolate(
                recon,
                size=(self.config.input_size, self.config.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {
            "recon": recon,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


if __name__ == "__main__":
    input_size = 256
    vae = SeismicSpatialVAE(input_size=input_size)
    input = torch.randn(4, 1, input_size, input_size)

    output = vae(input)

    print(output["recon"].shape)
    print(output["z"].shape)
    print(output["mu"].shape)
    print(output["logvar"].shape)
