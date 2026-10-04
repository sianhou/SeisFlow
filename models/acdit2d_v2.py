"""ACDiT V2: concatenate signal/coordinates once before shared reference encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from models.acdit2d import ACDiTCrossAttnBlock, ACDiTMLPBlock, ACDiTSelfAttnBlock
from models.pixeldit import (
    AugmentedDiTFinalLayer,
    ClassEmbedder,
    PatchTokenEmbedder,
    TimestepConditioner,
    precompute_freqs_cis_2d,
)
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS, BaseModelWrapper


class ACDiT2DModelV2(ModelMixin, ConfigMixin):
    """Encode concatenated inputs without per-block coordinate residuals."""

    @register_to_config
    def __init__(
            self,
            in_channels=66,
            out_channels=1,
            num_groups=12,
            hidden_size=768,
            depth=12,
            patch_size=2,
            num_classes=1000,
            max_period=10,
            upcast_attention=False,
            use_cross_attention=False,
    ):
        """Build one shared input projection and target/reference transformer layers.

        Args:
            in_channels: Total signal plus externally encoded coordinate channels.
                Targets and references use the same channel layout.
            out_channels: Predicted signal channels, excluding coordinates.
            num_groups: Attention heads; hidden_size / num_groups is divisible by four.
            hidden_size: Token and time/class conditioning width D.
            depth: Number of self-attention/cross-attention/MLP layers.
            patch_size: Square token patch side; input spatial sizes are divisible by it.
            num_classes: Class count, with one additional null-class embedding.
            max_period: Maximum period of the flow-time embedding.
            upcast_attention: Compute self/cross attention in float32 when True.
            use_cross_attention: Build reference cross-attention when True.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_groups = int(num_groups)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.max_period = float(max_period)
        self.upcast_attention = bool(upcast_attention)
        self.use_cross_attention = bool(use_cross_attention)
        self.patch_embedder = PatchTokenEmbedder(
            self.in_channels * self.patch_size ** 2, self.hidden_size,
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
        """Initialize the shared embedding and zero residual gates/output projections."""
        nn.init.xavier_uniform_(self.patch_embedder.proj.weight)
        nn.init.zeros_(self.patch_embedder.proj.bias)
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

    def embed_inputs(self, data):
        """Project concatenated channels once and obtain the local RoPE grid.

        Args:
            data: Signal/coordinate concatenation [B, in_channels, H, W].

        Returns:
            Tokens [B, N, D] and complex RoPE frequencies [N, head_dim // 2],
            where N=(H/p)*(W/p).
        """
        patches = F.unfold(
            data, kernel_size=self.patch_size, stride=self.patch_size,
        ).transpose(1, 2)
        grid = (data.shape[-2] // self.patch_size, data.shape[-1] // self.patch_size)
        if grid not in self.precompute_pos:
            self.precompute_pos[grid] = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, *grid,
            )
        return self.patch_embedder(patches), self.precompute_pos[grid].to(data.device)

    def forward(self, x, t, y, mask=None, r=None, cross_mask=None):
        """Predict target velocity from once-embedded target and reference inputs.

        Args:
            x: Noisy signal concatenated with coordinates [B, in_channels, H, W].
            t: Flow times [B], distinct from recording-time coordinates.
            y: Class indices [B].
            mask: Target self-attention allow/additive mask broadcastable to
                [B, heads, N, N]. References retain unmasked self-attention.
            r: Noised reference/coordinate concatenation [B, in_channels, Hr, Wr],
                or a nonempty list/tuple of such tensors. Each reference shares
                the target's embedder, self-attention and MLP, with its own local
                RoPE grid. None skips cross-attention; ignored when disabled.
            cross_mask: Allow/additive mask broadcastable to [B, heads, N, M],
                where M is the sum of reference token counts in reference order.

        Returns:
            Target velocity [B, out_channels, H, W]. Coordinates are neither
            separately projected nor re-added inside the transformer blocks.
        """
        batch_size, _, height, width = x.shape
        x_tokens, pos = self.embed_inputs(x)
        y_emb = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        t_emb = self.t_embedder(t.reshape(-1)).view(batch_size, 1, self.hidden_size)
        conditioning = F.silu(t_emb + y_emb)
        reference_inputs = []
        if self.use_cross_attention and r is not None:
            references = [r] if isinstance(r, torch.Tensor) else r
            reference_inputs = [self.embed_inputs(reference) for reference in references]
            r_pos = torch.cat([item[1] for item in reference_inputs], dim=0)

        for layer_index, block in enumerate(self.patch_blocks):
            x_tokens = block["attention"](x_tokens, conditioning, pos, mask)
            if reference_inputs:
                reference_inputs = [
                    (block["attention"](tokens, conditioning, ref_pos), ref_pos)
                    for tokens, ref_pos in reference_inputs
                ]
                reference_tokens = torch.cat([item[0] for item in reference_inputs], dim=1)
                x_tokens = block["cross_attention"](
                    x_tokens, reference_tokens, conditioning, pos, r_pos, cross_mask,
                )
            x_tokens = block["mlp"](x_tokens, conditioning)
            # The final reference MLP has no downstream consumer, as in V1.
            if layer_index < self.depth - 1:
                reference_inputs = [
                    (block["mlp"](tokens, conditioning), ref_pos)
                    for tokens, ref_pos in reference_inputs
                ]

        output_tokens = self.final_layer(x_tokens, conditioning)
        return F.fold(
            output_tokens.transpose(1, 2), output_size=(height, width),
            kernel_size=self.patch_size, stride=self.patch_size,
        )


class ACDiT2DWrapperV2(BaseModelWrapper):
    """Noise references and concatenate NeRF channels outside the V2 backbone."""

    model_cls = ACDiT2DModelV2

    def forward(self, x, timesteps, extra=None):
        """Concatenate coordinates after noising signals at the common flow time.

        Args:
            x: Noisy target signal [B, C_signal, H, W].
            timesteps: Flow times [B].
            extra: Mapping containing x_coord [B, C_nerf, H, W], optional label [B],
                mask and cross_mask. Enabled references use clean r, r_coord and
                independent Gaussian r_noise tensors, or matching lists/tuples.
                Reference signals/noises have shape [B, C_signal, Hr, Wr]; their
                encoded coordinates have shape [B, C_nerf, Hr, Wr]. Keep clean
                signals and noises fixed within an ODE run. Coordinates receive
                no noise; only signal channels follow (1-t)*noise + t*signal.

        Returns:
            Target velocity [B, out_channels, H, W], excluding coordinate channels.
        """
        references = None
        if self.model.use_cross_attention and extra.get("r") is not None:
            time = timesteps.reshape(-1, 1, 1, 1)
            if isinstance(extra["r"], torch.Tensor):
                noised = (1 - time) * extra["r_noise"] + time * extra["r"]
                references = torch.cat((noised, extra["r_coord"]), dim=1)
            else:
                references = [
                    torch.cat(((1 - time) * noise + time * reference, coords), dim=1)
                    for reference, noise, coords in zip(
                        extra["r"], extra["r_noise"], extra["r_coord"],
                    )
                ]
        inputs = torch.cat((x, extra["x_coord"]), dim=1)
        return self.model(
            inputs, timesteps, self.get_labels(x, extra),
            mask=extra.get("mask"), r=references, cross_mask=extra.get("cross_mask"),
        )


def build_acdit_2d_v2_wrapper(
        model_arch="T",
        in_channels=66,
        out_channels=1,
        patch_size=4,
        num_classes=1,
        max_period=10,
        upcast_attention=False,
        use_cross_attention=False,
        device=None,
):
    """Build an ACDiT V2 wrapper with a shared concatenated-input projection.

    Args:
        model_arch: Architecture preset from AUGMENTED_DIT_2D_CONFIGS.
        in_channels: Total signal plus encoded coordinate channels.
        out_channels: Predicted signal channels.
        patch_size: Square token patch side for targets and references.
        num_classes: Class count, plus one null-class embedding.
        max_period: Maximum flow-time embedding period.
        upcast_attention: Compute self/cross attention in float32 when True.
        use_cross_attention: Build and enable reference attention when True.
        device: Optional destination device.

    Returns:
        ACDiT2DWrapperV2 containing the configured V2 model.
    """
    model = ACDiT2DModelV2(
        **AUGMENTED_DIT_2D_CONFIGS[model_arch],
        in_channels=in_channels, out_channels=out_channels,
        patch_size=patch_size, num_classes=num_classes, max_period=max_period,
        upcast_attention=upcast_attention, use_cross_attention=use_cross_attention,
    )
    wrapper = ACDiT2DWrapperV2(model)
    return wrapper.to(device) if device is not None else wrapper
