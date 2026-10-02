"""Data self-attention with coordinate tokens injected before every block."""

import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from models.pixeldit import (
    AugmentedDiTFinalLayer,
    ClassEmbedder,
    FeedForward,
    PatchTokenEmbedder,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    precompute_freqs_cis_2d,
)


class ACDiTSelfAttentionBlock(nn.Module):
    """Conditioned self-attention with normalization and a gated residual."""

    def __init__(self, hidden_size, groups, upcast_attention=False):
        """Build the self-attention half of an augmented DiT block.

        Args:
            hidden_size: Token and conditioning width D.
            groups: Attention head count; head width must be divisible by four.
            upcast_attention: Compute attention in float32 when True.
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(
            hidden_size, num_heads=groups, qkv_bias=False,
            upcast_attention=upcast_attention,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, x, c, pos, mask=None):
        """Return tokens after conditioned self-attention and residual addition.

        Args:
            x: Input tokens [B, N, D].
            c: Activated time/class conditioning [B, 1, D].
            pos: Complex RoPE frequencies [N, head_dim // 2].
            mask: Boolean allow mask or additive mask broadcastable to
                [B, groups, N, N]. None permits all token pairs.

        Returns:
            Updated tokens [B, N, D].
        """
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        normalized = apply_adaln(self.norm(x), shift, scale)
        return x + gate * self.attn(normalized, pos, mask=mask)


class ACDiTMLPBlock(nn.Module):
    """Conditioned feed-forward network with normalization and a gated residual."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        """Build the feed-forward half of an augmented DiT block.

        Args:
            hidden_size: Token and conditioning width D.
            mlp_ratio: Feed-forward hidden width multiplier.
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        """Return tokens after the conditioned MLP and residual addition.

        Args:
            x: Input tokens [B, N, D].
            c: Activated time/class conditioning [B, 1, D].

        Returns:
            Updated tokens [B, N, D].
        """
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        normalized = apply_adaln(self.norm(x), shift, scale)
        return x + gate * self.mlp(normalized)


class ACDiT2DModel(ModelMixin, ConfigMixin):
    """Reinforce coordinate conditioning at every data self-attention block."""

    @register_to_config
    def __init__(
            self,
            in_channels=4,
            in_coords_channels=5,
            out_channels=None,
            num_groups=12,
            hidden_size=768,
            depth=12,
            patch_size=2,
            num_classes=1000,
            max_period=10,
            upcast_attention=False,
    ):
        """Build data/coordinate embedders and independent attention/MLP pairs.

        Args:
            in_channels: Input data channels.
            in_coords_channels: Coordinate feature channels supplied by the caller,
                usually SX/SY/RX/RY/T.
            out_channels: Prediction channels; None uses in_channels.
            num_groups: Attention heads; hidden_size / num_groups must be divisible
                by four for the shared two-dimensional RoPE implementation.
            hidden_size: Token width shared by data and coordinate projections.
            depth: Number of attention/MLP pairs; coordinates enter before attention.
            patch_size: Nonoverlapping square token patch side; each input's height
                and width must be divisible by it.
            num_classes: Class count, with one additional null-class embedding.
            max_period: Maximum period of the flow-time embedding.
            upcast_attention: Compute self-attention in float32 when True.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.in_coords_channels = int(in_coords_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.num_groups = int(num_groups)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.max_period = float(max_period)
        self.upcast_attention = bool(upcast_attention)

        self.x_patch_embedder = PatchTokenEmbedder(
            self.in_channels * self.patch_size ** 2, self.hidden_size,
        )
        self.coords_patch_embedder = PatchTokenEmbedder(
            self.in_coords_channels * self.patch_size ** 2, self.hidden_size,
        )
        self.t_embedder = TimestepConditioner(
            self.hidden_size, max_period=self.max_period,
        )
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)
        self.patch_blocks = nn.ModuleList([
            nn.ModuleDict({
                "attention": ACDiTSelfAttentionBlock(
                    self.hidden_size, self.num_groups,
                    upcast_attention=self.upcast_attention,
                ),
                "mlp": ACDiTMLPBlock(self.hidden_size),
            })
            for _ in range(self.depth)
        ])
        self.final_layer = AugmentedDiTFinalLayer(
            self.hidden_size, self.patch_size, self.out_channels,
        )
        self.precompute_pos = {}
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize embeddings and zero the residual modulation/output projections."""
        for embedder in (self.x_patch_embedder, self.coords_patch_embedder):
            nn.init.xavier_uniform_(embedder.proj.weight)
            nn.init.zeros_(embedder.proj.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.patch_blocks:
            for component in block.values():
                nn.init.zeros_(component.adaLN_modulation[0].weight)
                nn.init.zeros_(component.adaLN_modulation[0].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[0].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[0].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def fetch_pos(self, height, width, device):
        """Return cached local RoPE frequencies for one patch grid.

        Args:
            height: Number of patch rows.
            width: Number of patch columns.
            device: Device on which to return the frequencies.

        Returns:
            Complex frequencies [height * width, head_dim // 2].
        """
        key = (height, width)
        if key not in self.precompute_pos:
            self.precompute_pos[key] = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, height, width,
            )
        return self.precompute_pos[key].to(device)

    def embed_inputs(self, data, coords):
        """Encode data and coordinates separately for repeated coordinate injection.

        Args:
            data: Input data [B, in_channels, H, W].
            coords: Aligned coordinate features [B, in_coords_channels, H, W].

        Returns:
            Data tokens [B, N, D], coordinate tokens [B, N, D], and local
            RoPE frequencies [N, head_dim // 2], with N=(H/p)*(W/p).
        """
        data_patches = F.unfold(
            data, kernel_size=self.patch_size, stride=self.patch_size,
        ).transpose(1, 2)
        coord_patches = F.unfold(
            coords, kernel_size=self.patch_size, stride=self.patch_size,
        ).transpose(1, 2)
        x_tokens = self.x_patch_embedder(data_patches)
        coord_tokens = self.coords_patch_embedder(coord_patches)
        pos = self.fetch_pos(
            data.shape[-2] // self.patch_size, data.shape[-1] // self.patch_size,
            data.device,
        )
        return x_tokens, coord_tokens, pos

    def forward(self, x, x_coord, t, y, mask=None):
        """Predict data by adding coordinates before each attention/MLP pair.

        Args:
            x: Noisy input data [B, in_channels, H, W].
            x_coord: Aligned coordinate features [B, in_coords_channels, H, W].
                Embedded once and reused at each block entrance.
            t: Flow times [B], distinct from recording-time coordinates.
            y: Class indices [B].
            mask: Boolean (True permits) or additive self-attention mask
                broadcastable to [B, heads, N, N], where N=(H/p)*(W/p).

        Returns:
            Velocity/data prediction [B, out_channels, H, W].
        """
        batch_size, _, height, width = x.shape
        x_tokens, coord_tokens, pos = self.embed_inputs(x, x_coord)
        y_emb = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        t_emb = self.t_embedder(t.reshape(-1)).view(batch_size, 1, self.hidden_size)
        conditioning = F.silu(t_emb + y_emb)

        for block in self.patch_blocks:
            x_tokens = x_tokens + coord_tokens
            x_tokens = block["attention"](x_tokens, conditioning, pos, mask)
            x_tokens = block["mlp"](x_tokens, conditioning)

        output_tokens = self.final_layer(x_tokens, conditioning)
        return F.fold(
            output_tokens.transpose(1, 2), output_size=(height, width),
            kernel_size=self.patch_size, stride=self.patch_size,
        )
