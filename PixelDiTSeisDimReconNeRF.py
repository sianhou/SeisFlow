import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.dataset import PairedPatchDataset, PatchDataset
from core.sampler import Sampler
from core.trainer import Trainer
from models.dinov2 import DINOv2
from models.nerf import get_nerf_conditioning_channels, encode_nerf_conditioning
from models.wrapper import (
    Pixel_DiT_2D_CONFIGS,
    PixelDiT2DWrapper,
    build_pixeldit_2d_wrapper,
)


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


class PixelDiTSeisDimReconNeRFTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.dino = None
        self.repa_projection = None
        self.repa_projected_feature = None
        self.repa_clean_images = None

    @property
    def repa_enabled(self):
        return self.args.repa_lambda > 0.0

    def setup_dataset(self):
        return PairedPatchDataset(
            self.args.input_dir,
            self.args.input_dim_dir,
        )

    def setup_model(self):
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        nerf_dim_channels = get_nerf_conditioning_channels(raw_dim_channels, self.args)
        model = build_pixeldit_2d_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1 + nerf_dim_channels,
            out_channels=1,
            num_classes=1,
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

            dino_hidden_size = int(self.dino.encoder.embed_dim)
            model.configure_repa(
                align_layer=self.args.repa_align_layer,
                projection_dim=dino_hidden_size,
            )
            self.repa_projection = model.repa_projection

        return model

    def preprocess_batch(self, batch):
        clean_images, conditioning = batch
        clean_images = clean_images.to(self.device, non_blocking=True)
        conditioning = conditioning.to(self.device, non_blocking=True)
        conditioning = encode_nerf_conditioning(conditioning, self.args)
        self.repa_clean_images = clean_images
        self.repa_projected_feature = None
        return clean_images, {"concat_conditioning": conditioning}

    def compute_loss(self, model_output, sample, mode="velocity"):
        if self.repa_enabled:
            if not isinstance(model_output, tuple) or len(model_output) != 2:
                raise RuntimeError(
                    "REPA-enabled PixelDiT must return "
                    "(prediction, projected_feature)."
                )
            prediction, self.repa_projected_feature = model_output
        else:
            prediction = model_output
            self.repa_projected_feature = None
        return super().compute_loss(prediction, sample, mode=mode)

    def compute_auxiliary_loss(self):
        if not self.repa_enabled:
            return 0
        if self.repa_projected_feature is None:
            raise RuntimeError(
                "REPA projected feature was not returned by the model."
            )

        src_feature = self.repa_projected_feature
        dino_input = (self.repa_clean_images + 1.0) / 2.0
        dino_input = dino_input[:, 0:1].repeat(1, 3, 1, 1)
        dino_input = dino_input.clamp(0.0, 1.0)

        with torch.inference_mode():
            dst_feature = self.dino(dino_input).float()

        src_feature, dst_feature = self.match_repa_tokens(
            src_feature,
            dst_feature,
        )
        repa_loss = 1.0 - F.cosine_similarity(
            src_feature,
            dst_feature,
            dim=-1,
        ).mean()
        self.repa_projected_feature = None
        return self.args.repa_lambda * repa_loss

    @staticmethod
    def match_repa_tokens(src_feature, dst_feature):
        if src_feature.shape[1] == dst_feature.shape[1]:
            return src_feature, dst_feature

        batch_size, src_tokens, channels = src_feature.shape
        dst_tokens = dst_feature.shape[1]
        src_size = int(src_tokens ** 0.5)
        dst_size = int(dst_tokens ** 0.5)
        if src_size * src_size != src_tokens or dst_size * dst_size != dst_tokens:
            raise ValueError(
                "REPA token counts must form square grids when resizing: "
                f"src={src_tokens}, dst={dst_tokens}."
            )
        src_spatial = src_feature.view(
            batch_size, src_size, src_size, channels
        ).permute(0, 3, 1, 2)
        if dst_tokens < src_tokens:
            src_spatial = F.adaptive_avg_pool2d(
                src_spatial,
                (dst_size, dst_size),
            )
        else:
            src_spatial = F.interpolate(
                src_spatial,
                size=(dst_size, dst_size),
                mode="bilinear",
                align_corners=False,
            )
        src_feature = src_spatial.permute(0, 2, 3, 1).reshape(
            batch_size, dst_tokens, channels
        )
        return src_feature, dst_feature

    def from_pretrained(self):
        super().from_pretrained()
        if not self.repa_enabled or not self.args.ckpt:
            return
        projection_path = Path(self.args.ckpt) / "repa_projection.pth"
        if not projection_path.is_file():
            raise FileNotFoundError(
                f"REPA checkpoint is missing projection weights: {projection_path}"
            )
        state = torch.load(
            projection_path,
            map_location=self.device,
            weights_only=True,
        )
        self.repa_projection.load_state_dict(state)

    def save_pretrained(self, epoch):
        super().save_pretrained(epoch)
        if not self.is_main_process or not self.repa_enabled:
            return
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch:05d}"
        torch.save(
            self.repa_projection.state_dict(),
            checkpoint_path / "repa_projection.pth",
        )


class PixelDiTSeisDimReconNeRFSampler(Sampler):
    def setup_dataset(self):
        return PatchDataset(self.args.input_dim_dir)

    def setup_model(self):
        return PixelDiT2DWrapper.from_pretrained(
            save_directory=self.args.ckpt,
            device=self.device,
            use_ema=self.args.use_ema,
        )

    def preprocess_batch(self, batch):
        batch = np.array(batch, copy=True)
        if batch.ndim == 3:
            batch = batch[:, np.newaxis, :, :]
        conditioning = torch.from_numpy(batch).float()
        conditioning = conditioning.to(self.device, non_blocking=True)
        conditioning = encode_nerf_conditioning(conditioning, self.args)
        noise = torch.randn(
            (
                conditioning.shape[0],
                1,
                conditioning.shape[-2],
                conditioning.shape[-1],
            ),
            device=self.device,
            dtype=conditioning.dtype,
        )
        return noise, {"concat_conditioning": conditioning}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train or sample a distributed PixelDiT flow-matching model "
            "from noise to seismic patches with NeRF-encoded dimension coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 PixelDiTSeisDimReconNeRF.py "
            "--input_dir ./dataset256/train "
            "--input_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_dim_recon "
            "--model_arch T --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Sample:\n"
            "  torchrun --nproc_per_node=4 PixelDiTSeisDimReconNeRF.py sample "
            "--ckpt ./output_dim_recon/run/checkpoint_epoch_01000 "
            "--input_dim_dir ./dataset256/sample_dim "
            "--output_dir ./seisdimrecon_output "
            "--batch_size 32 --solver_step_size 0.05 --device cuda"
        ),
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "sample"],
        default="train",
        help="Run mode. Omit for training.",
    )
    parser.add_argument("--input_dir", default="./dataset/train")
    parser.add_argument("--input_dim_dir", default="./dataset/train_dim")
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument(
        "--model_arch",
        choices=sorted(Pixel_DiT_2D_CONFIGS.keys()),
        default="T",
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--solver_step_size", default=0.05, type=float)
    parser.add_argument("--clip_recon", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument(
        "--upcast_attention",
        action="store_true",
        help="Compute PixelDiT scaled dot-product attention in float32 under AMP.",
    )
    parser.add_argument(
        "--repa_lambda",
        default=0.0,
        type=float,
        help="REPA cosine-loss weight; zero disables DINO supervision.",
    )
    parser.add_argument(
        "--repa_align_layer",
        default=8,
        type=int,
        help="1-based PixelDiT patch block used for REPA features.",
    )
    parser.add_argument(
        "--dino_model_name",
        default="dinov2_vitb14",
        choices=(
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ),
    )
    parser.add_argument(
        "--dino_hub_dir",
        default="./dinov2_cache",
        help="Local Torch Hub directory for the DINOv2 teacher.",
    )
    parser.add_argument(
        "--nerf_bands",
        default=6,
        type=int,
        help="Number of exponential Fourier frequency bands used to encode dimension-coordinate channels.",
    )
    parser.add_argument(
        "--no-nerf_include_input",
        dest="nerf_include_input",
        action="store_false",
        default=True,
        help="Do not include original dimension-coordinate channels; use only Fourier sin/cos channels.",
    )
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "linear"],
        default="constant",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--save_every_epochs", default=50, type=int)
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Maintain EMA weights and use them for sampling when loading a checkpoint.",
    )
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--ema_warmup", default=0, type=int)
    parser.add_argument("--log_id", default=None)
    parser.add_argument("--log_console", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--world_size", default=1, type=int)
    return parser


def run(args):
    if args.mode == "sample":
        return PixelDiTSeisDimReconNeRFSampler(args).run()
    return PixelDiTSeisDimReconNeRFTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
