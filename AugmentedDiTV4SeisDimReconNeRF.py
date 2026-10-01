"""Train and sample V4 using NeRF coordinates as spatial AdaLN conditions."""

from AugmentedDiTSeisDimReconNeRF import (
    AugmentedDiTSeisDimReconNeRFSampler as BaseSampler,
    AugmentedDiTSeisDimReconNeRFTrainer as BaseTrainer,
    build_parser as build_base_parser,
)
from models.augmented_dit_2d_v4 import (
    AugmentedDiT2DWrapperV4,
    build_augmented_dit_2d_wrapper_v4,
)
from models.dinov2 import DINOv2
from models.nerf import get_nerf_conditioning_channels


class AugmentedDiTSeisDimReconNeRFTrainer(BaseTrainer):
    """Reuse the original training workflow with a separate coordinate condition."""

    def setup_model(self):
        """Return a single-channel V4 wrapper, configuring optional REPA supervision."""
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        model = build_augmented_dit_2d_wrapper_v4(
            model_arch=self.args.model_arch,
            in_channels=1,
            coord_channels=get_nerf_conditioning_channels(raw_dim_channels, self.args),
            out_channels=1,
            patch_size=self.args.patch_size,
            max_period=self.args.max_period,
            upcast_attention=self.args.upcast_attention,
            device=self.device,
        )
        if self.repa_enabled:
            self.dino = DINOv2(
                model_name=self.args.dino_model_name,
                hub_dir=self.args.dino_hub_dir,
            ).to(self.device)
            self.dino.eval()
            for parameter in self.dino.parameters():
                parameter.requires_grad_(False)
            model.configure_repa(
                align_layer=self.args.repa_align_layer,
                projection_dim=int(self.dino.encoder.embed_dim),
            )
            self.repa_projection = model.repa_projection
        return model

    def preprocess_batch(self, batch):
        """Reuse NeRF encoding while routing coordinates into y instead of data channels.

        Args:
            batch: Clean signal [B, 1, H, W] and coordinates [B, C_dim, H, W].

        Returns:
            Clean signal and {'y': encoded_coordinates}, with y shaped
            [B, coord_channels, H, W]. The base trainer adds noise only to data.
        """
        clean_images, extra = super().preprocess_batch(batch)
        return clean_images, {"y": extra["concat_conditioning"]}


class AugmentedDiTSeisDimReconNeRFSampler(BaseSampler):
    """Reuse file-based sampling with V4 checkpoints and continuous coordinate labels."""

    def setup_model(self):
        """Return the saved V4 wrapper, optionally applying checkpoint EMA weights."""
        return AugmentedDiT2DWrapperV4.from_pretrained(
            save_directory=self.args.ckpt,
            device=self.device,
            use_ema=self.args.use_ema,
        )

    def preprocess_batch(self, batch):
        """Prepare single-channel noise and route encoded coordinates into y.

        Args:
            batch: Numpy coordinates [B, C_dim, H, W] or [B, H, W].

        Returns:
            Noise [B, 1, H, W] and {'y': encoded_coordinates} matching the
            training interface. Coordinates remain fixed during ODE sampling.
        """
        noise, extra = super().preprocess_batch(batch)
        return noise, {"y": extra["concat_conditioning"]}


def build_parser():
    """Return the existing train/sample CLI with V4 descriptions and examples."""
    parser = build_base_parser()
    parser.description = (
        "Train or sample AugmentedDiT V4 with single-channel seismic data and "
        "NeRF coordinates supplied separately as patch-wise AdaLN conditions."
    )
    parser.epilog = parser.epilog.replace(
        "AugmentedDiTSeisDimReconNeRF.py", "AugmentedDiTV4SeisDimReconNeRF.py",
    )
    return parser


def run(args):
    """Execute V4 training or sampling using the existing distributed workflow.

    Args:
        args: CLI namespace returned by build_parser().parse_args().
    """
    if args.mode == "sample":
        AugmentedDiTSeisDimReconNeRFSampler(args).run()
    else:
        AugmentedDiTSeisDimReconNeRFTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
