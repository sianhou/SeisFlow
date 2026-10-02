"""Coordinate-conditioned data attention with optional reference cross-attention."""

import torch
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
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS, BaseModelWrapper


class ACDiTSelfAttnBlock(nn.Module):
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


class ACDiTCrossAttnBlock(nn.Module):
    """Conditioned cross-attention from input tokens to reference tokens."""

    def __init__(self, hidden_size, groups, upcast_attention=False):
        """Build reference attention with normalization and a gated residual.

        Args:
            hidden_size: Input, reference and conditioning width D.
            groups: Attention head count; head width must be divisible by four.
            upcast_attention: Compute attention in float32 when True.
        """
        super().__init__()
        self.num_heads = groups
        self.head_dim = hidden_size // groups
        self.upcast_attention = upcast_attention
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.ref_norm = RMSNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, x, ref, c, pos, ref_pos, mask=None):
        """Return input tokens updated by reading reference keys and values.

        Args:
            x: Input tokens [B, N, D], providing queries and the residual.
            ref: Reference tokens [B, M, D], providing keys and values.
                References receive normalization but no time/class modulation.
            c: Activated input time/class conditioning [B, 1, D].
            pos: Input RoPE frequencies [N, head_dim // 2].
            ref_pos: Reference RoPE frequencies [M, head_dim // 2].
            mask: Boolean allow mask or additive mask broadcastable to
                [B, groups, N, M]. None permits all input/reference pairs.

        Returns:
            Updated input tokens [B, N, D]. Reference tokens are not updated.
        """
        batch_size, token_count, hidden_size = x.shape
        ref_count = ref.shape[1]
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        normalized = apply_adaln(self.norm(x), shift, scale)
        reference = self.ref_norm(ref)
        q = self.q_proj(normalized).reshape(
            batch_size, token_count, self.num_heads, self.head_dim,
        )
        k = self.k_proj(reference).reshape(
            batch_size, ref_count, self.num_heads, self.head_dim,
        )
        v = self.v_proj(reference).reshape(
            batch_size, ref_count, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, _ = apply_rotary_emb(q, q, pos)
        k, _ = apply_rotary_emb(k, k, ref_pos)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        if self.upcast_attention:
            upcast_mask = mask
            if mask is not None and mask.dtype != torch.bool:
                upcast_mask = mask.float()
            with torch.autocast(device_type=x.device.type, enabled=False):
                attended = F.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(), attn_mask=upcast_mask,
                    dropout_p=0.0,
                )
            attended = attended.to(v.dtype)
        else:
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0,
            )
        attended = attended.transpose(1, 2).reshape(
            batch_size, token_count, hidden_size,
        )
        return x + gate * self.proj(attended)


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
    """Inject coordinates and read reference data at each transformer layer."""

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
            use_cross_attention=False,
    ):
        """Build embedders and layers of self-attention, cross-attention and MLP.

        Args:
            in_channels: Data channels shared by input and references.
            in_coords_channels: Coordinate feature channels supplied by the caller,
                usually SX/SY/RX/RY/T.
            out_channels: Prediction channels; None uses in_channels.
            num_groups: Attention heads; hidden_size / num_groups must be divisible
                by four for the shared two-dimensional RoPE implementation.
            hidden_size: Token width shared by data and coordinate projections.
            depth: Number of self-attention/cross-attention/MLP layers;
                input coordinates enter before self-attention.
            patch_size: Nonoverlapping square token patch side; each input's height
                and width must be divisible by it.
            num_classes: Class count, with one additional null-class embedding.
            max_period: Maximum period of the flow-time embedding.
            upcast_attention: Compute self/cross attention in float32 when True.
            use_cross_attention: Build and enable reference attention when True.
                Defaults to False; reference inputs are ignored when disabled.
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
        self.use_cross_attention = bool(use_cross_attention)

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
        self.patch_blocks = nn.ModuleList()
        for _ in range(self.depth):
            block = nn.ModuleDict({
                "attention": ACDiTSelfAttnBlock(
                    self.hidden_size, self.num_groups,
                    upcast_attention=self.upcast_attention,
                ),
            })
            if self.use_cross_attention:
                block["cross_attention"] = ACDiTCrossAttnBlock(
                    self.hidden_size, self.num_groups,
                    upcast_attention=self.upcast_attention,
                )
            block["mlp"] = ACDiTMLPBlock(self.hidden_size)
            self.patch_blocks.append(block)
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
            data: Input or reference data [B, in_channels, H, W].
            coords: Aligned input or reference coordinates [B, in_coords_channels, H, W].

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

    def forward(self, x, x_coord, t, y, mask=None, r=None, r_coord=None, cross_mask=None):
        """Predict data with per-layer coordinate injection and reference attention.

        Args:
            x: Noisy input data [B, in_channels, H, W].
            x_coord: Aligned coordinate features [B, in_coords_channels, H, W].
                Embedded once and reused at each block entrance.
            t: Flow times [B], distinct from recording-time coordinates.
            y: Class indices [B].
            mask: Boolean (True permits) or additive self-attention mask
                broadcastable to [B, heads, N, N], where N=(H/p)*(W/p).
            r: Reference data [B, in_channels, Hr, Wr], or a nonempty list/tuple
                of such tensors. Each spatial dimension must be divisible by p.
                References use the same data/coordinate embedders as x. Their summed
                tokens are reused by every cross-attention layer without passing
                through self-attention or MLP blocks.
                None skips cross-attention; ignored when use_cross_attention=False.
            r_coord: Reference coordinates [B, in_coords_channels, Hr, Wr], or a
                list/tuple aligned with r. Required when reference attention runs;
                ignored when use_cross_attention=False or r is None.
            cross_mask: Boolean allow mask or additive cross-attention mask
                broadcastable to [B, heads, N, M], independent of mask. M is the
                total reference token count, concatenated in reference order.

        Returns:
            Velocity/data prediction [B, out_channels, H, W].
        """
        batch_size, _, height, width = x.shape
        x_tokens, x_coord_tokens, pos = self.embed_inputs(x, x_coord)
        y_emb = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        t_emb = self.t_embedder(t.reshape(-1)).view(batch_size, 1, self.hidden_size)
        conditioning = F.silu(t_emb + y_emb)
        if self.use_cross_attention and r is not None:
            if isinstance(r, torch.Tensor):
                r_tokens, r_coord_tokens, r_pos = self.embed_inputs(r, r_coord)
            else:
                reference_inputs = [
                    self.embed_inputs(reference, coords)
                    for reference, coords in zip(r, r_coord)
                ]
                r_tokens = torch.cat([item[0] for item in reference_inputs], dim=1)
                r_coord_tokens = torch.cat([item[1] for item in reference_inputs], dim=1)
                r_pos = torch.cat([item[2] for item in reference_inputs], dim=0)
            reference_tokens = r_tokens + r_coord_tokens

        for block in self.patch_blocks:
            x_tokens = x_tokens + x_coord_tokens
            x_tokens = block["attention"](x_tokens, conditioning, pos, mask)
            if self.use_cross_attention and r is not None:
                x_tokens = block["cross_attention"](
                    x_tokens, reference_tokens, conditioning, pos, r_pos, cross_mask,
                )
            x_tokens = block["mlp"](x_tokens, conditioning)

        output_tokens = self.final_layer(x_tokens, conditioning)
        return F.fold(
            output_tokens.transpose(1, 2), output_size=(height, width),
            kernel_size=self.patch_size, stride=self.patch_size,
        )


class ACDiT2DWrapper(BaseModelWrapper):
    """Adapt ACDiT to the common flow-matching and checkpoint interface."""

    model_cls = ACDiT2DModel

    def forward(self, x, timesteps, extra=None):
        """Predict velocities with separate input and reference coordinates.

        Args:
            x: Noisy data [B, in_channels, H, W].
            timesteps: Flow-matching times [B].
            extra: Mapping with required x_coord [B, in_coords_channels, H, W],
                optional label [B], and optional r/r_coord tensors or lists of
                tensors [B, C, Hr, Wr]. Coordinates are already NeRF encoded.
                Optional mask and cross_mask are forwarded to the model.

        Returns:
            Velocity prediction [B, out_channels, H, W].
        """
        return self.model(
            x, extra["x_coord"], timesteps, self.get_labels(x, extra),
            mask=extra.get("mask"), r=extra.get("r"), r_coord=extra.get("r_coord"),
            cross_mask=extra.get("cross_mask"),
        )


def build_acdit_2d_wrapper(
        model_arch="T",
        in_channels=1,
        in_coords_channels=65,
        out_channels=1,
        patch_size=4,
        num_classes=1,
        max_period=10,
        upcast_attention=False,
        use_cross_attention=False,
        device=None,
):
    """Build an ACDiT wrapper using the existing transformer architecture presets.

    Args:
        model_arch: Architecture preset name in AUGMENTED_DIT_2D_CONFIGS.
        in_channels: Data channels shared by input and references.
        in_coords_channels: Coordinate channels after external NeRF encoding.
        out_channels: Number of predicted data channels.
        patch_size: Square token patch side for input and reference data.
        num_classes: Class count, with one additional null-class embedding.
        max_period: Maximum flow-time embedding period.
        upcast_attention: Compute attention in float32 when True.
        use_cross_attention: Build and enable reference attention when True.
        device: Optional destination device for the wrapper.

    Returns:
        ACDiT2DWrapper containing the configured model.
    """
    model = ACDiT2DModel(
        **AUGMENTED_DIT_2D_CONFIGS[model_arch],
        in_channels=in_channels, in_coords_channels=in_coords_channels,
        out_channels=out_channels, patch_size=patch_size, num_classes=num_classes,
        max_period=max_period, upcast_attention=upcast_attention,
        use_cross_attention=use_cross_attention,
    )
    wrapper = ACDiT2DWrapper(model)
    return wrapper.to(device) if device is not None else wrapper
