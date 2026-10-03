"""Train and reconstruct with ACDiT and random references from the input dataset."""

import argparse

import numpy as np
import torch

from core.dataset import PairedPatchDataset, PatchDataset
from core.sampler import Sampler
from core.trainer import Trainer
from models.acdit2d import ACDiT2DWrapper, build_acdit_2d_wrapper
from models.nerf import encode_nerf_conditioning, get_nerf_conditioning_channels
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show argument defaults while retaining multiline CLI examples."""


def sample_reference_conditioning(dataset, batch_size, args, device):
    """Draw independent reference pairs from the full dataset with replacement.

    Args:
        dataset: PairedPatchDataset containing signal [1, H, W] and matching
            raw coordinates [C_dim, H, W] at each index.
        batch_size: Number of targets, each receiving args.use_ref references.
        args: CLI namespace with use_ref, nerf_bands and nerf_include_input.
        device: Device receiving reference tensors and NeRF coordinate features.

    Returns:
        Mapping with r and r_coord lists of length args.use_ref. Their entries
        have shapes [B, 1, H, W] and [B, C_nerf, H, W]. References may repeat
        or equal the target; only the data/coordinate pair shares an index.
    """
    indices = torch.randint(len(dataset), (args.use_ref, batch_size))
    references, reference_coordinates = [], []
    for reference_indices in indices.tolist():
        pairs = [dataset[index] for index in reference_indices]
        data = torch.stack([pair[0] for pair in pairs]).to(device, non_blocking=True)
        coords = torch.stack([pair[1] for pair in pairs]).to(device, non_blocking=True)
        references.append(data)
        reference_coordinates.append(encode_nerf_conditioning(coords, args))
    return {"r": references, "r_coord": reference_coordinates}


class ACDiTSeisDimReconNeRFTrainer(Trainer):
    """Train target velocities with optional random clean reference patches."""

    def setup_dataset(self):
        """Return the paired input dataset used for both targets and references."""
        return PairedPatchDataset(self.args.input_dir, self.args.input_dim_dir)

    def setup_model(self):
        """Return an ACDiT wrapper with cross-attention enabled by use_ref >= 1."""
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        return build_acdit_2d_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1,
            in_coords_channels=get_nerf_conditioning_channels(raw_dim_channels, self.args),
            out_channels=1,
            patch_size=self.args.patch_size,
            num_classes=1,
            max_period=self.args.max_period,
            upcast_attention=self.args.upcast_attention,
            use_cross_attention=self.args.use_ref >= 1,
            device=self.device,
        )

    def preprocess_batch(self, batch):
        """Prepare target coordinates and independently sampled reference pairs.

        Args:
            batch: Signal [B, 1, H, W] and raw coordinates [B, C_dim, H, W]
                from the paired input DataLoader.

        Returns:
            Clean targets [B, 1, H, W] and a conditioning mapping containing
            x_coord [B, C_nerf, H, W], plus r/r_coord lists when use_ref >= 1.
            The base trainer adds noise only to the targets, leaving references clean.
        """
        clean_images, coordinates = batch
        clean_images = clean_images.to(self.device, non_blocking=True)
        coordinates = coordinates.to(self.device, non_blocking=True)
        extra = {"x_coord": encode_nerf_conditioning(coordinates, self.args)}
        if self.args.use_ref >= 1:
            extra.update(sample_reference_conditioning(
                self.dataset, clean_images.shape[0], self.args, self.device,
            ))
        return clean_images, extra


class ACDiTSeisDimReconNeRFSampler(Sampler):
    """Reconstruct target coordinates using an independent training reference pool."""

    def setup_dataset(self):
        """Return target coordinates and open paired training references separately.

        input_dim_dir defines the reconstruction targets. When use_ref >= 1,
        ref_dir/ref_dim_dir supply training signals and their paired coordinates.
        No target signal data is read. Without references only input_dim_dir is needed.
        """
        if self.args.use_ref >= 1:
            self.reference_dataset = PairedPatchDataset(
                self.args.ref_dir, self.args.ref_dim_dir,
            )
        return PatchDataset(self.args.input_dim_dir)

    def setup_model(self):
        """Return the ACDiT checkpoint wrapper, optionally applying saved EMA weights."""
        return ACDiT2DWrapper.from_pretrained(
            save_directory=self.args.ckpt, device=self.device, use_ema=self.args.use_ema,
        )

    def preprocess_batch(self, batch):
        """Prepare initial noise, target coordinates and fixed references for an ODE run.

        Args:
            batch: Numpy coordinates [B, C_dim, H, W] or [B, H, W].

        Returns:
            Noise [B, 1, H, W] and conditioning with encoded x_coord and optional
            r/r_coord lists. References are sampled here once per batch and stay
            fixed for every ODE evaluation of that batch.
        """
        coordinates = np.array(batch, copy=True)
        if coordinates.ndim == 3:
            coordinates = coordinates[:, np.newaxis, :, :]
        coordinates = torch.from_numpy(coordinates).float().to(self.device, non_blocking=True)
        noise = torch.randn(
            coordinates.shape[0], 1, coordinates.shape[-2], coordinates.shape[-1],
            device=self.device, dtype=coordinates.dtype,
        )
        extra = {"x_coord": encode_nerf_conditioning(coordinates, self.args)}
        if self.args.use_ref >= 1:
            extra.update(sample_reference_conditioning(
                self.reference_dataset, noise.shape[0], self.args, self.device,
            ))
        return noise, extra


def build_parser():
    """Return the ACDiT train/sample CLI; use_ref controls reference count and cross-attention."""
    parser = argparse.ArgumentParser(
        description="Train or sample ACDiT with NeRF coordinates and random paired references.",
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 ACDiTSeisDimReconNeRF.py "
            "--input_dir ./dataset/train --input_dim_dir ./dataset/train_dim "
            "--use_ref 2 --output_dir ./output_acdit --model_arch T --device cuda\n\n"
            "  Sample:\n"
            "  python ACDiTSeisDimReconNeRF.py sample "
            "--ckpt ./output_acdit/run/checkpoint_epoch_01000 "
            "--input_dim_dir ./dataset/valid_dim "
            "--ref_dir ./dataset/train --ref_dim_dir ./dataset/train_dim "
            "--use_ref 2 --output_dir ./recon_acdit --device cuda\n\n"
            "Each target receives use_ref independent random references from the paired "
            "training dataset, with replacement and without excluding itself. Sampling "
            "targets come from input_dim_dir; references come from ref_dir/ref_dim_dir. Signal and "
            "coordinate files must have matching patch ordering. Both coordinate branches "
            "use the same NeRF settings. Sampling references stay fixed within each ODE run."
        ),
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", choices=["train", "sample"], default="train")
    parser.add_argument("--input_dir", default="./dataset/train")
    parser.add_argument("--input_dim_dir", default="./dataset/train_dim")
    parser.add_argument(
        "--ref_dir", default="./dataset/train",
        help="Training signal reference pool for sampling; unused during training or with use_ref=0.",
    )
    parser.add_argument(
        "--ref_dim_dir", default="./dataset/train_dim",
        help="Coordinates paired with ref_dir for sampling, separate from target input_dim_dir.",
    )
    parser.add_argument(
        "--use_ref", default=0, type=int,
        help="Random references per target: 0 disables cross-attention; >=1 enables it.",
    )
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
    """Execute ACDiT training or file-based reconstruction.

    Args:
        args: CLI namespace returned by build_parser().parse_args().
    """
    if args.mode == "sample":
        ACDiTSeisDimReconNeRFSampler(args).run()
    else:
        ACDiTSeisDimReconNeRFTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
