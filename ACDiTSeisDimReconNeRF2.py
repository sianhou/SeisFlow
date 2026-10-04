"""Train/sample ACDiT V2 with input-only signal/NeRF coordinate concatenation."""

from ACDiTSeisDimReconNeRF import (
    ACDiTSeisDimReconNeRFSampler,
    ACDiTSeisDimReconNeRFTrainer,
    build_parser as build_acdit_parser,
)
from models.acdit2d_v2 import ACDiT2DWrapperV2, build_acdit_2d_v2_wrapper
from models.nerf import get_nerf_conditioning_channels


class ACDiTSeisDimReconNeRF2Trainer(ACDiTSeisDimReconNeRFTrainer):
    """Reuse aligned references/flow noise while training the input-concat backbone."""

    def setup_model(self):
        """Return V2 with signal plus NeRF input channels and signal-only output."""
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        return build_acdit_2d_v2_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1 + get_nerf_conditioning_channels(raw_dim_channels, self.args),
            out_channels=1,
            patch_size=self.args.patch_size,
            num_classes=1,
            max_period=self.args.max_period,
            upcast_attention=self.args.upcast_attention,
            use_cross_attention=bool(self.dataset.references),
            device=self.device,
        )


class ACDiTSeisDimReconNeRF2Sampler(ACDiTSeisDimReconNeRFSampler):
    """Reuse aligned reference sampling with a V2 checkpoint and fixed ODE noise."""

    def setup_model(self):
        """Return the V2 checkpoint wrapper, optionally loading its EMA weights."""
        return ACDiT2DWrapperV2.from_pretrained(
            save_directory=self.args.ckpt, device=self.device, use_ema=self.args.use_ema,
        )


def build_parser():
    """Return the aligned-reference CLI with V2-specific descriptions and examples."""
    parser = build_acdit_parser()
    parser.description = (
        "Train or sample ACDiT V2: concatenate each signal with its NeRF coordinates "
        "once at the network input, without per-block coordinate residuals."
    )
    parser.epilog = parser.epilog.replace(
        "ACDiTSeisDimReconNeRF.py", "ACDiTSeisDimReconNeRF2.py",
    ).replace("output_acdit", "output_acdit_v2").replace("recon_acdit", "recon_acdit_v2")
    return parser


def run(args):
    """Execute V2 training or reconstruction with the parsed CLI namespace.

    Args:
        args: Namespace returned by build_parser().parse_args().
    """
    if args.mode == "sample":
        ACDiTSeisDimReconNeRF2Sampler(args).run()
    else:
        ACDiTSeisDimReconNeRF2Trainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
