"""Augmented DiT with continuous coordinate conditions applied through patch-wise AdaLN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from models.pixeldit import (
    AugmentedDiTBlock,
    AugmentedDiTFinalLayer,
    PatchTokenEmbedder,
    TimestepConditioner,
    precompute_freqs_cis_2d,
)
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS, AugmentedDiT2DWrapper


class AugmentedDiT2DModelV4(ModelMixin, ConfigMixin):
    """Encode data and coordinates separately, conditioning each patch through AdaLN."""

    @register_to_config
    def __init__(
            self,
            in_channels=1,
            coord_channels=65,
            out_channels=None,
            num_groups=12,
            hidden_size=768,
            depth=12,
            patch_size=2,
            max_period=10,
            upcast_attention=False,
    ):
        """Build the existing AugmentedDiT backbone with a coordinate patch encoder.

        Args:
            in_channels: Data channels in x; excludes coordinate channels.
            coord_channels: Channels in y after external coordinate encoding.
                The default 65 represents five coordinates with six Fourier
                bands and the original coordinates included.
            out_channels: Prediction channels; None uses in_channels.
            num_groups: Number of attention heads.
            hidden_size: Token width; each attention head must have a width
                divisible by four for the existing 2D rotary embeddings.
            depth: Number of AugmentedDiT blocks.
            patch_size: Square patch side length shared by data and coordinates.
            max_period: Maximum period used by the timestep embedding.
            upcast_attention: Whether attention is computed in float32.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.coord_channels = int(coord_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.num_groups = int(num_groups)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.patch_size = int(patch_size)
        self.max_period = float(max_period)
        self.upcast_attention = bool(upcast_attention)

        self.patch_embedder = PatchTokenEmbedder(
            self.in_channels * self.patch_size ** 2, self.hidden_size,
        )
        self.y_embedder = PatchTokenEmbedder(
            self.coord_channels * self.patch_size ** 2, self.hidden_size,
        )
        self.t_embedder = TimestepConditioner(
            self.hidden_size, max_period=self.max_period,
        )
        self.patch_blocks = nn.ModuleList([
            AugmentedDiTBlock(
                self.hidden_size,
                self.num_groups,
                upcast_attention=self.upcast_attention,
            )
            for _ in range(self.depth)
        ])
        self.final_layer = AugmentedDiTFinalLayer(
            self.hidden_size, self.patch_size, self.out_channels,
        )
        self.precompute_pos = {}
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize both patch encoders and retain the backbone's zero AdaLN/output initialization."""
        for embedder in (self.patch_embedder, self.y_embedder):
            nn.init.xavier_uniform_(embedder.proj.weight)
            nn.init.zeros_(embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[0].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[0].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def fetch_pos(self, height, width, device):
        """Return cached rotary embeddings for the local patch grid.

        Args:
            height: Number of patch rows.
            width: Number of patch columns.
            device: Device on which to return the embeddings.

        Returns:
            Complex rotary frequencies [height * width, head_dim // 2].
        """
        key = (height, width)
        if key not in self.precompute_pos:
            self.precompute_pos[key] = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, height, width,
            )
        return self.precompute_pos[key].to(device)

    def forward(self, x, t, y, mask=None, return_patch_feature_at=None):
        """Predict data using continuous coordinates as spatial AdaLN conditions.

        Args:
            x: Noisy data [B, in_channels, H, W], without coordinate channels.
            t: Flow-matching times [B], distinct from physical recording time.
            y: Float coordinates [B, coord_channels, H, W], already encoded
                by the caller when using NeRF features. Coordinates and data
                must share spatial ordering and dimensions divisible by
                patch_size. No coordinate pooling or integer label lookup occurs.
            mask: Optional attention mask broadcastable to [B, num_groups, N, N],
                where N = (H // patch_size) * (W // patch_size).
            return_patch_feature_at: Optional zero-based block index whose
                output tokens should be returned alongside the prediction.

        Returns:
            Prediction [B, out_channels, H, W], or (prediction, patch_feature)
            with patch_feature [B, N, hidden_size] when a block is requested.
        """
        batch_size, _, height, width = x.shape
        pos = self.fetch_pos(
            height // self.patch_size, width // self.patch_size, x.device,
        )
        tokens = F.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size,
        ).transpose(1, 2)
        tokens = self.patch_embedder(tokens)
        coordinate_tokens = F.unfold(
            y, kernel_size=self.patch_size, stride=self.patch_size,
        ).transpose(1, 2)
        y_emb = self.y_embedder(coordinate_tokens)
        t_emb = self.t_embedder(t.reshape(-1)).view(batch_size, 1, self.hidden_size)
        conditioning = F.silu(t_emb + y_emb)

        patch_feature = None
        for block_index, block in enumerate(self.patch_blocks):
            tokens = block(tokens, conditioning, pos, mask)
            if block_index == return_patch_feature_at:
                patch_feature = tokens

        output_tokens = self.final_layer(tokens, conditioning)
        output = F.fold(
            output_tokens.transpose(1, 2).contiguous(),
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        if return_patch_feature_at is not None:
            if patch_feature is None:
                raise ValueError(
                    "Requested patch feature layer is out of range: "
                    f"index={return_patch_feature_at}, "
                    f"depth={len(self.patch_blocks)}."
                )
            return output, patch_feature
        return output


class AugmentedDiT2DWrapperV4(AugmentedDiT2DWrapper):
    """Reuse checkpoint and optional REPA support with continuous coordinates as y."""

    model_cls = AugmentedDiT2DModelV4

    def forward(self, x, timesteps, extra=None):
        """Pass coordinates separately from the noisy data to the V4 model.

        Args:
            x: Noisy data [B, in_channels, H, W], without coordinate channels.
            timesteps: Flow-matching times [B].
            extra: Mapping containing required y [B, coord_channels, H, W]
                and an optional mask broadcastable to [B, num_groups, N, N].
                N is the patch count; y contains externally encoded coordinates.

        Returns:
            Prediction [B, out_channels, H, W], or (prediction, projected_feature)
            with features [B, N, projection_dim] when REPA is configured.
        """
        output = self.model(
            x, timesteps, y=extra["y"], mask=extra.get("mask"),
            return_patch_feature_at=self.repa_align_index,
        )
        if self.repa_projection is None:
            return output
        prediction, patch_feature = output
        with torch.autocast(device_type=patch_feature.device.type, enabled=False):
            projected_feature = self.repa_projection(patch_feature.float())
        return prediction, projected_feature


def build_augmented_dit_2d_wrapper_v4(
        model_arch="T",
        in_channels=1,
        coord_channels=65,
        out_channels=None,
        num_groups=None,
        hidden_size=None,
        depth=None,
        patch_size=2,
        max_period=10,
        upcast_attention=False,
        device=None,
):
    """Build a V4 wrapper from an existing AugmentedDiT architecture preset.

    Args:
        model_arch: Preset name: Nano, T, S, L, or XL.
        in_channels: Data channels only, excluding coordinates.
        coord_channels: Channels in y after external coordinate encoding.
        out_channels: Prediction channels; None uses in_channels.
        num_groups: Attention head count; None uses the preset value.
        hidden_size: Token width; None uses the preset value.
        depth: Transformer block count; None uses the preset value.
        patch_size: Square patch side length shared by data and coordinates.
        max_period: Maximum period used by the timestep embedding.
        upcast_attention: Whether attention is computed in float32.
        device: Optional destination device for model parameters.

    Returns:
        AugmentedDiT2DWrapperV4 containing the configured V4 model.
    """
    architecture = AUGMENTED_DIT_2D_CONFIGS[model_arch]
    model = AugmentedDiT2DModelV4(
        in_channels=in_channels,
        coord_channels=coord_channels,
        out_channels=out_channels,
        num_groups=architecture["num_groups"] if num_groups is None else num_groups,
        hidden_size=architecture["hidden_size"] if hidden_size is None else hidden_size,
        depth=architecture["depth"] if depth is None else depth,
        patch_size=patch_size,
        max_period=max_period,
        upcast_attention=upcast_attention,
    )
    wrapper = AugmentedDiT2DWrapperV4(model)
    return wrapper.to(device) if device is not None else wrapper
