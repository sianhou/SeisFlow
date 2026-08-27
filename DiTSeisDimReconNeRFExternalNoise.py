import argparse
from pathlib import Path

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


class DiTSeisDimReconNeRFExternalNoiseTrainer(Trainer):
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


class DiTSeisDimReconNeRFExternalNoiseSampler(Sampler):
    def setup_dataset(self):
        self.noise_data_path = (
            None if self.args.noise_data_dir is None else Path(self.args.noise_data_dir)
        )
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

        if self.noise_data_path is None:
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
        else:
            noise_batch = next(self.noise_batch_iterator)
            noise_batch = np.array(noise_batch, copy=True)
            if noise_batch.ndim == 3:
                noise_batch = noise_batch[:, np.newaxis, :, :]
            noise = torch.from_numpy(noise_batch).float()
            noise = noise.to(self.device, non_blocking=True)
        return noise, {"concat_conditioning": conditioning}

    def iter_noise_batches(self):
        for input_file in self.rank_files:
            noise_file = self.noise_data_path / Path(input_file).name
            noise_array = np.load(noise_file, mmap_mode="r")
            for batch_start in range(0, len(noise_array), self.args.batch_size):
                batch_end = min(batch_start + self.args.batch_size, len(noise_array))
                yield noise_array[batch_start:batch_end]

    def sample_one_epoch(self):
        if self.noise_data_path is not None:
            self.noise_batch_iterator = iter(self.iter_noise_batches())
        return super().sample_one_epoch()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train or sample DiTSeisDimReconNeRFExternalNoise "
            "from noise to seismic patches with NeRF-encoded dimension coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 DiTSeisDimReconNeRFExternalNoise.py "
            "--input_dir ./dataset256/train "
            "--input_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_dim_recon "
            "--model_arch DiT_T_4 --input_size 256 --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Sample:\n"
            "  torchrun --nproc_per_node=4 DiTSeisDimReconNeRFExternalNoise.py sample "
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
    parser.add_argument(
        "--noise_data_dir",
        default=None,
        help="Optional directory containing external noise patch files.",
    )
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
        return DiTSeisDimReconNeRFExternalNoiseSampler(args).run()
    return DiTSeisDimReconNeRFExternalNoiseTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
