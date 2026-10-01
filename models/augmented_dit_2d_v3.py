import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from torch.nn.functional import scaled_dot_product_attention

from models.pixeldit import apply_rotary_emb, PatchTokenEmbedder, TimestepConditioner, ClassEmbedder, RMSNorm, \
    RotaryAttention, FeedForward, apply_adaln, precompute_freqs_cis_2d, AugmentedDiTFinalLayer
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS, BaseModelWrapper


# 问题2:
# 参考 token 数被强制等于目标 token 数。从 x 取得 N，随后用同一个 N reshape k、v。
# 只要参考 token 数 M≠N，就会报错。我用目标4个、参考2个 token 运行，得到 reshape 错误。
# 即使现在恰好一对同尺寸 patch 能运行，多个参考样本拼接后仍会失败。

# 问题3:
# Q 和 K 使用同一份位置编码。将同一个 pos 同时用于目标 Q 和参考 K。
# 两者长度不同会产生形状问题；即使长度相同，也相当于默认它们位于完全相同的网格位置。
# 参考样本若有独立位置，需要分别传入 pos_q、pos_k，或者明确只使用各自局部位置并另加相对几何偏置。


class RotaryCrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = RMSNorm,
            upcast_attention: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.upcast_attention = bool(upcast_attention)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(k).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(v).reshape(B, N, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        if self.upcast_attention:
            output_dtype = v.dtype
            upcast_mask = mask
            if upcast_mask is not None and upcast_mask.dtype != torch.bool:
                upcast_mask = upcast_mask.float()
            with torch.autocast(device_type=q.device.type, enabled=False):
                x = scaled_dot_product_attention(
                    q.float(),
                    k.float(),
                    v.float(),
                    attn_mask=upcast_mask,
                    dropout_p=0.0,
                )
            x = x.to(dtype=output_dtype)
        else:
            x = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AugmentedDiTBlockV2(nn.Module):
    def __init__(
            self,
            hidden_size,
            groups,
            mlp_ratio=4.0,
            adaLN_modulation=None,
            upcast_attention=False,
            use_cross_attention=False
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(
            hidden_size,
            num_heads=groups,
            qkv_bias=False,
            upcast_attention=upcast_attention,
        )

        if use_cross_attention:
            self.cross_norm = RMSNorm(hidden_size, eps=1e-6)
            self.cross_attn = RotaryCrossAttention(
                hidden_size,
                num_heads=groups,
                qkv_bias=False,
                upcast_attention=upcast_attention,
            )

        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        if not self.use_cross_attention:
            self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
                nn.Linear(hidden_size, 9 * hidden_size, bias=True)
            )

    def forward(self, x, c, pos, k=None, v=None, mask=None):
        if not self.use_cross_attention:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp \
                = self.adaLN_modulation(c).chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp \
                = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        if self.use_cross_attention:
            x = x + gate_mca * self.cross_attn(apply_adaln(self.cross_norm(x), shift_mca, scale_mca), k, v, pos,
                                               mask=mask)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AugmentedDiT2DModelV3(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            in_channels=4,
            out_channels=None,
            num_groups=12,
            hidden_size=768,
            depth=12,
            patch_size=2,
            num_classes=1000,
            max_period=10,
            upcast_attention=False,
            use_cross_attention=False
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.num_groups = int(num_groups)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.max_period = float(max_period)
        self.upcast_attention = bool(upcast_attention)
        self.use_cross_attention = bool(use_cross_attention)

        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if self.hidden_size % self.num_groups != 0:
            raise ValueError("hidden_size must be divisible by num_groups.")
        head_dim = self.hidden_size // self.num_groups
        if head_dim % 4 != 0:
            raise ValueError(
                "2D rotary embeddings require hidden_size / num_groups "
                f"to be divisible by 4, got head_dim={head_dim}."
            )
        if self.max_period <= 0:
            raise ValueError(f"max_period must be positive, got {max_period}.")

        patch_channels = self.in_channels * self.patch_size ** 2
        self.patch_embedder = PatchTokenEmbedder(
            patch_channels,
            self.hidden_size,
            bias=True,
        )

        self.t_embedder = TimestepConditioner(
            self.hidden_size,
            max_period=self.max_period,
        )

        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)

        self.patch_blocks = nn.ModuleList(
            [
                AugmentedDiTBlockV2(
                    self.hidden_size,
                    self.num_groups,
                    upcast_attention=self.upcast_attention,
                    use_cross_attention=self.use_cross_attention,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_layer = AugmentedDiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )
        self.precompute_pos = dict()
        self.initialize_weights()

    def fetch_pos(self, height, width, device):
        key = (height, width)
        if key not in self.precompute_pos:
            self.precompute_pos[key] = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups,
                height,
                width,
            )
        return self.precompute_pos[key].to(device)

    def initialize_weights(self):
        weight = self.patch_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.zeros_(self.patch_embedder.proj.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[0].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[0].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
            self,
            x,
            t,
            y,
            r=None,
            mask=None,
            return_patch_feature_at=None,
    ):
        if x.dim() != 4:
            raise ValueError("AugmentedDiT2DModel expects x with shape [B,C,H,W].")
        batch_size, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}."
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "Input height and width must be divisible by patch_size; "
                f"got {(height, width)} and patch_size={self.patch_size}."
            )

        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        pos = self.fetch_pos(patch_height, patch_width, x.device)
        tokens = torch.nn.functional.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        tokens = self.patch_embedder(tokens)

        if self.use_cross_attention:
            ref_tokens = torch.nn.functional.unfold(
                r,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).transpose(1, 2)
            ref_tokens = self.patch_embedder(ref_tokens)

        t_emb = self.t_embedder(t.reshape(-1)).view(batch_size, 1, self.hidden_size)
        y_emb = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        conditioning = nn.functional.silu(t_emb + y_emb)

        patch_feature = None
        for block_index, block in enumerate(self.patch_blocks):
            if not self.use_cross_attention:
                tokens = block(x=tokens, c=conditioning, pos=pos, mask=mask)
            else:
                tokens = block(x=tokens, c=conditioning, pos=pos, k=ref_tokens, v=ref_tokens, mask=mask)
            if block_index == return_patch_feature_at:
                patch_feature = tokens

        output_tokens = self.final_layer(tokens, conditioning)
        output_tokens = output_tokens.transpose(1, 2).contiguous()
        output = torch.nn.functional.fold(
            output_tokens,
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


class AugmentedDiT2DWrapperV3(BaseModelWrapper):
    """Adapt V3 conditioning and inherit model, training-state, and EMA persistence."""

    model_cls = AugmentedDiT2DModelV3

    def forward(self, x, timesteps, extra=None):
        """Predict the target velocity with an optional coordinate-aware reference.

        Args:
            x: Target state [B, C_data, H, W], or an already concatenated input.
            timesteps: Flow-matching times [B].
            extra: Optional mapping containing concat_conditioning [B, C_cond,
                H, W], label [B], r [B, in_channels, H, W], and an attention
                mask broadcastable to [B, num_groups, N, N]. N is the patch
                count. Reference r already contains data and encoded reference
                coordinates in the same channel order as the target input.
                Labels default to zero; cross attention requires r.

        Returns:
            Prediction tensor [B, out_channels, H, W].
        """
        extra = {} if extra is None else extra
        x = self.concat_conditioning(x, extra)
        labels = self.get_labels(x, extra)
        return self.model(x, timesteps, labels, r=extra.get("r"), mask=extra.get("mask"))


def build_augmented_dit_2d_wrapper_v3(
        model_arch="T",
        in_channels=4,
        out_channels=None,
        num_groups=None,
        hidden_size=None,
        depth=None,
        patch_size=2,
        num_classes=1000,
        max_period=10,
        upcast_attention=False,
        use_cross_attention=False,
        device=None,
):
    """Build a V3 wrapper using the shared AugmentedDiT architecture presets.

    Args:
        model_arch: Preset name: Nano, T, S, L, or XL.
        in_channels: Data plus encoded-coordinate channels in both input branches.
        out_channels: Prediction channels; None uses in_channels.
        num_groups: Attention head count; None uses the preset value.
        hidden_size: Token width; None uses the preset value.
        depth: Transformer block count; None uses the preset value.
        patch_size: Side length of each square patch in both input branches.
        num_classes: Class count, excluding the additional null-class embedding.
        max_period: Maximum period used by the timestep embedding.
        upcast_attention: Whether attention is computed in float32.
        use_cross_attention: Whether to use the complete reference in extra['r'].
        device: Optional destination device for model parameters.

    Returns:
        AugmentedDiT2DWrapperV3 containing an AugmentedDiT2DModelV3.
    """
    architecture = AUGMENTED_DIT_2D_CONFIGS[model_arch]
    model = AugmentedDiT2DModelV3(
        in_channels=in_channels,
        out_channels=out_channels,
        num_groups=architecture["num_groups"] if num_groups is None else num_groups,
        hidden_size=architecture["hidden_size"] if hidden_size is None else hidden_size,
        depth=architecture["depth"] if depth is None else depth,
        patch_size=patch_size,
        num_classes=num_classes,
        max_period=max_period,
        upcast_attention=upcast_attention,
        use_cross_attention=use_cross_attention,
    )
    wrapper = AugmentedDiT2DWrapperV3(model)
    return wrapper.to(device) if device is not None else wrapper
