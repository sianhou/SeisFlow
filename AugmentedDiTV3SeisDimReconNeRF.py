"""Train and sample V3 with NeRF coordinates on both target and reference patches."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from core.dataset import PairedPatchDataset, PatchDataset
from core.sampler import Sampler
from core.trainer import Trainer
from models.augmented_dit_2d_v3 import (
    AUGMENTED_DIT_2D_CONFIGS,
    AugmentedDiT2DWrapperV3,
    build_augmented_dit_2d_wrapper_v3,
)
from models.nerf import encode_nerf_conditioning, get_nerf_conditioning_channels


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show argument defaults while retaining multiline CLI examples."""


class ReferencedPatchDataset(Dataset):
    """Load aligned target and reference patches with their own coordinates."""

    def __init__(self, seismic_dir, dim_dir, ref_dir, ref_dim_dir):
        """Index four NPY directories with matching file and patch ordering.

        Args:
            seismic_dir: Target signal directory, with one channel per patch.
            dim_dir: Target coordinate directory, with C_dim channels per patch.
            ref_dir: Reference signal directory, with one channel per patch.
            ref_dim_dir: Reference coordinates with the same channel definitions
                and normalization as dim_dir. All patches share spatial shape.
        """
        self.paired = PairedPatchDataset(seismic_dir, dim_dir)
        self.dataset1 = self.paired.dataset1
        self.reference = PairedPatchDataset(ref_dir, ref_dim_dir)

    def __len__(self):
        """Return the number of aligned target/reference patch pairs."""
        return len(self.paired)

    def __getitem__(self, index):
        """Return target, target coordinates, reference, and reference coordinates.

        Args:
            index: Global patch index shared by all four directories.

        Returns:
            Four tensors shaped [1, H, W], [C_dim, H, W], [1, H, W],
            and [C_dim, H, W], respectively.
        """
        seismic, dimensions = self.paired[index]
        reference, reference_dimensions = self.reference[index]
        return seismic, dimensions, reference, reference_dimensions


class AugmentedDiTSeisDimReconNeRFTrainer(Trainer):
    """Train V3 velocities using clean reference signals and encoded coordinates."""

    def setup_dataset(self):
        """Return paired target patches, including reference pairs when configured."""
        if self.args.ref:
            ref_dim_dir = self.args.ref_dim or f"{Path(self.args.ref)}_dim"
            return ReferencedPatchDataset(
                self.args.input_dir, self.args.input_dim_dir,
                self.args.ref, ref_dim_dir,
            )
        return PairedPatchDataset(self.args.input_dir, self.args.input_dim_dir)

    def setup_model(self):
        """Return a V3 wrapper predicting one signal channel from encoded inputs."""
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        nerf_dim_channels = get_nerf_conditioning_channels(raw_dim_channels, self.args)
        return build_augmented_dit_2d_wrapper_v3(
            model_arch=self.args.model_arch,
            in_channels=1 + nerf_dim_channels,
            out_channels=1,
            patch_size=self.args.patch_size,
            num_classes=1,
            max_period=self.args.max_period,
            upcast_attention=self.args.upcast_attention,
            use_cross_attention=bool(self.args.ref),
            device=self.device,
        )

    def preprocess_batch(self, batch):
        """Encode both coordinate sets and concatenate reference data and coordinates.

        Args:
            batch: Target signal [B, 1, H, W] and coordinates [B, C_dim, H, W],
                followed by reference signal and coordinates when ref is enabled.

        Returns:
            Clean target [B, 1, H, W] and extra conditioning. The base trainer
            adds noise only to the target. extra['r'] contains the clean reference
            and its encoded coordinates in the target input's channel order.
        """
        if self.args.ref:
            clean_images, conditioning, reference, reference_coordinates = batch
        else:
            clean_images, conditioning = batch
        clean_images = clean_images.to(self.device, non_blocking=True)
        conditioning = conditioning.to(self.device, non_blocking=True)
        extra = {"concat_conditioning": encode_nerf_conditioning(conditioning, self.args)}
        if self.args.ref:
            reference = reference.to(self.device, non_blocking=True)
            reference_coordinates = reference_coordinates.to(self.device, non_blocking=True)
            encoded_reference = encode_nerf_conditioning(reference_coordinates, self.args)
            extra["r"] = torch.cat((reference, encoded_reference), dim=1)
        return clean_images, extra


class AugmentedDiTSeisDimReconNeRFSampler(Sampler):
    """Sample V3 signals from target coordinates and optional reference pairs."""

    def __init__(self, args):
        """Initialize sampling and the memory-mapped reference-file cache.

        Args:
            args: Parsed CLI namespace containing checkpoint and input directories.
        """
        super().__init__(args)
        self._reference_file = None
        self._reference_array = None
        self._reference_coordinates = None

    def setup_dataset(self):
        """Return the target coordinate dataset whose files define sampling output."""
        return PatchDataset(self.args.input_dim_dir)

    def setup_model(self):
        """Return the V3 checkpoint wrapper, optionally loading its EMA weights."""
        return AugmentedDiT2DWrapperV3.from_pretrained(
            save_directory=self.args.ckpt,
            device=self.device,
            use_ema=self.args.use_ema,
        )

    def load_input_batch(self, input_array, input_file, batch_start, batch_end):
        """Read matching target coordinates, reference signals, and reference coordinates.

        Args:
            input_array: NPY target coordinates [P, C_dim, H, W] or [P, H, W].
            input_file: Target coordinate filename, shared by both reference files.
            batch_start: Inclusive patch index within the current file.
            batch_end: Exclusive patch index within the current file.

        Returns:
            Target coordinate batch, or a tuple with reference data and coordinates.
        """
        dimensions = input_array[batch_start:batch_end]
        if not self.args.ref:
            return dimensions
        reference_file = Path(self.args.ref) / Path(input_file).name
        if self._reference_file != reference_file:
            ref_dim_dir = self.args.ref_dim or f"{Path(self.args.ref)}_dim"
            self._reference_array = np.load(reference_file, mmap_mode="r")
            self._reference_coordinates = np.load(
                Path(ref_dim_dir) / Path(input_file).name, mmap_mode="r",
            )
            self._reference_file = reference_file
        return (
            dimensions,
            self._reference_array[batch_start:batch_end],
            self._reference_coordinates[batch_start:batch_end],
        )

    def preprocess_batch(self, batch):
        """Create initial noise and prepare the same conditioning used in training.

        Args:
            batch: Target coordinate array, or a tuple of target coordinates,
                reference data, and reference coordinates. Arrays have shape
                [B, C, H, W] or [B, H, W] for a single channel.

        Returns:
            Gaussian noise [B, 1, H, W] and an extra mapping containing encoded
            target coordinates and the complete reference input when configured.
        """
        if self.args.ref:
            batch, reference, reference_coordinates = batch
        conditioning = encode_nerf_conditioning(self.to_tensor(batch), self.args)
        noise = torch.randn(
            conditioning.shape[0], 1, conditioning.shape[-2], conditioning.shape[-1],
            device=self.device, dtype=conditioning.dtype,
        )
        extra = {"concat_conditioning": conditioning}
        if self.args.ref:
            reference = self.to_tensor(reference)
            reference_coordinates = encode_nerf_conditioning(
                self.to_tensor(reference_coordinates), self.args,
            )
            extra["r"] = torch.cat((reference, reference_coordinates), dim=1)
        return noise, extra

    def to_tensor(self, array):
        """Copy an NPY batch into a float32 tensor on the sampling device.

        Args:
            array: Signal or coordinate batch [B, C, H, W] or [B, H, W].

        Returns:
            Tensor [B, C, H, W], inserting a single channel for 3D arrays.
        """
        array = np.array(array, copy=True)
        if array.ndim == 3:
            array = array[:, np.newaxis, :, :]
        return torch.from_numpy(array).float().to(self.device, non_blocking=True)


def build_parser():
    """Return the CLI parser for V3 training and sampling with coordinate-aware references."""
    parser = argparse.ArgumentParser(
        description=(
            "Train or sample AugmentedDiT V3 with a shared patch embedder for "
            "target and reference signals plus NeRF-encoded coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 AugmentedDiTV3SeisDimReconNeRF.py "
            "--input_dir ./dataset256/train --input_dim_dir ./dataset256/train_dim "
            "--ref ./dataset256/train_ref --ref_dim ./dataset256/train_ref_dim "
            "--output_dir ./output_v3 --model_arch T --device cuda\n\n"
            "  Sample:\n"
            "  python AugmentedDiTV3SeisDimReconNeRF.py sample "
            "--ckpt ./output_v3/run/checkpoint_epoch_01000 "
            "--input_dim_dir ./dataset256/valid_dim "
            "--ref ./dataset256/valid_ref --ref_dim ./dataset256/valid_ref_dim "
            "--output_dir ./recon_v3 --device cuda\n\n"
            "Target and reference files must share filenames, patch ordering, spatial "
            "shape, and coordinate channel definitions. Use the same coordinate "
            "normalization and NeRF options for training and sampling."
        ),
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", choices=["train", "sample"], default="train")
    parser.add_argument("--input_dir", default="./dataset/train")
    parser.add_argument("--input_dim_dir", default="./dataset/train_dim")
    parser.add_argument("--ref", default=None, help="Reference signal patch directory; enables cross attention.")
    parser.add_argument("--ref_dim", default=None, help="Reference coordinate directory; defaults to <ref>_dim.")
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--model_arch", choices=sorted(AUGMENTED_DIT_2D_CONFIGS), default="T")
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--max_period", default=10, type=float)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--solver_step_size", default=0.05, type=float)
    parser.add_argument("--clip_recon", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--upcast_attention", action="store_true", help="Compute attention in float32 under AMP.")
    parser.add_argument("--nerf_bands", default=6, type=int, help="Number of Fourier frequency bands for both coordinate sets.")
    parser.add_argument(
        "--no-nerf_include_input", dest="nerf_include_input", action="store_false",
        default=True, help="Encode both coordinate sets with Fourier sin/cos channels only.",
    )
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--lr_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--save_every_epochs", default=50, type=int)
    parser.add_argument(
        "--use_ema", action=argparse.BooleanOptionalAction, default=True,
        help="Maintain EMA weights and use them when sampling checkpoints.",
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
    """Run V3 training or sampling to completion.

    Args:
        args: CLI namespace returned by build_parser().parse_args().
    """
    if args.mode == "sample":
        AugmentedDiTSeisDimReconNeRFSampler(args).run()
    else:
        AugmentedDiTSeisDimReconNeRFTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
