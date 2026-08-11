import argparse

import numpy as np
import torch

from core.dataset import PairedPatchDataset, PatchDataset
from core.sampler import Sampler
from core.trainer import Trainer
from models.nerf import get_nerf_conditioning_channels, encode_nerf_conditioning
from models.wrapper import (
    DIT_TRANSFORMER_2D_CONFIGS,
    DiTTransformer2DWrapper,
    build_dit_transformer_2d_wrapper,
)


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


class DiTSeisDimReconNeRFTrainer(Trainer):
    def setup_dataset(self):
        return PairedPatchDataset(
            self.args.input_dir,
            self.args.input_dim_dir,
        )

    def setup_model(self):
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        nerf_dim_channels = get_nerf_conditioning_channels(raw_dim_channels, self.args)
        return build_dit_transformer_2d_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1 + nerf_dim_channels,
            out_channels=1,
            sample_size=self.args.input_size,
            num_embeds_ada_norm=1,
            upcast_attention=self.args.upcast_attention,
            device=self.device,
        )

    def preprocess_batch(self, batch):
        clean_images, conditioning = batch
        clean_images = clean_images.to(self.device, non_blocking=True)
        conditioning = conditioning.to(self.device, non_blocking=True)
        conditioning = encode_nerf_conditioning(conditioning, self.args)
        return clean_images, {"concat_conditioning": conditioning}


class DiTSeisDimReconNeRFSampler(Sampler):
    def setup_dataset(self):
        return PatchDataset(self.args.input_dim_dir)

    def setup_model(self):
        return DiTTransformer2DWrapper.from_pretrained(
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
            "Train or sample a distributed DiT flow-matching model "
            "from noise to seismic patches with NeRF-encoded dimension coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 DiTSeisDimReconNeRF.py "
            "--input_dir ./dataset256/train "
            "--input_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_dim_recon "
            "--model_arch DiT_T_4 --input_size 256 --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Sample:\n"
            "  torchrun --nproc_per_node=4 DiTSeisDimReconNeRF.py sample "
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
        choices=sorted(DIT_TRANSFORMER_2D_CONFIGS.keys()),
        default="DiT_T_4",
    )
    parser.add_argument("--input_size", default=64, type=int)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--solver_step_size", default=0.05, type=float)
    parser.add_argument("--clip_recon", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--upcast_attention", action="store_true")
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
        return DiTSeisDimReconNeRFSampler(args).run()
    return DiTSeisDimReconNeRFTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
