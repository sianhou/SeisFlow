"""Train and reconstruct with ACDiT and aligned, independently noised references."""

import argparse
from pathlib import Path

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


def reference_directories(args):
    """Return configured (signal, coordinate) directory pairs in reference-slot order.

    Args:
        args: CLI namespace containing ref_dir1/ref_dim_dir1 and ref_dir2/ref_dim_dir2.

    Returns:
        Zero, one or two directory pairs; an omitted signal directory disables its slot.
    """
    return [
        (data_dir, coord_dir)
        for data_dir, coord_dir in (
            (args.ref_dir1, args.ref_dim_dir1),
            (args.ref_dir2, args.ref_dim_dir2),
        )
        if data_dir is not None
    ]


class AlignedReferencePatchDataset(PairedPatchDataset):
    """Read target and reference pairs together so shuffling preserves alignment."""

    def __init__(self, input_dir, input_dim_dir, reference_dirs):
        """Open paired targets and references with matching filenames and patch ordering.

        Args:
            input_dir: Target signal directory.
            input_dim_dir: Target coordinate directory.
            reference_dirs: Ordered (signal, coordinate) directory pairs. Each
                directory has the same filenames and per-file patch counts as targets.
        """
        super().__init__(input_dir, input_dim_dir)
        self.references = [PairedPatchDataset(*paths) for paths in reference_dirs]

    def __getitem__(self, index):
        """Return a target and its reference pairs at the same global patch index.

        Args:
            index: Index into the common sorted file/patch ordering.

        Returns:
            Target [1, H, W], coordinates [C_dim, H, W], and a list of reference
            (signal, coordinate) pairs with the same respective shapes.
        """
        data, coords = super().__getitem__(index)
        return data, coords, [reference[index] for reference in self.references]


def prepare_reference_conditioning(reference_pairs, args, device):
    """Encode aligned references and draw one independent noise tensor for each.

    Args:
        reference_pairs: Batched tensor pairs ([B, 1, H, W], [B, C_dim, H, W]).
        args: CLI namespace containing nerf_bands and nerf_include_input.
        device: Destination device for conditioning tensors.

    Returns:
        Empty mapping without references, otherwise r/r_coord/r_noise lists.
        Clean r and fixed r_noise have shape [B, 1, H, W]; r_coord has shape
        [B, C_nerf, H, W]. The wrapper mixes r and r_noise at the current flow time.
    """
    if not reference_pairs:
        return {}
    references, reference_coordinates, reference_noise = [], [], []
    for data, coords in reference_pairs:
        data = data.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        references.append(data)
        reference_coordinates.append(encode_nerf_conditioning(coords, args))
        reference_noise.append(torch.randn_like(data))
    return {"r": references, "r_coord": reference_coordinates, "r_noise": reference_noise}


class ACDiTSeisDimReconNeRFTrainer(Trainer):
    """Train target velocities with optional aligned reference patches."""

    def setup_dataset(self):
        """Return targets and configured references in a shared patch ordering."""
        return AlignedReferencePatchDataset(
            self.args.input_dir, self.args.input_dim_dir, reference_directories(self.args),
        )

    def setup_model(self):
        """Return an ACDiT wrapper with cross-attention when reference directories are given."""
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
            use_cross_attention=bool(self.dataset.references),
            device=self.device,
        )

    def preprocess_batch(self, batch):
        """Prepare aligned reference conditions before the trainer samples target flow times.

        Args:
            batch: Target [B, 1, H, W], coordinates [B, C_dim, H, W], and
                a list of batched reference signal/coordinate pairs from the DataLoader.

        Returns:
            Clean targets [B, 1, H, W] and a conditioning mapping containing
            x_coord [B, C_nerf, H, W], plus r/r_coord/r_noise lists when enabled.
            The wrapper noises references at the same flow time as the target.
        """
        clean_images, coordinates, reference_pairs = batch
        clean_images = clean_images.to(self.device, non_blocking=True)
        coordinates = coordinates.to(self.device, non_blocking=True)
        extra = {"x_coord": encode_nerf_conditioning(coordinates, self.args)}
        extra.update(prepare_reference_conditioning(reference_pairs, self.args, self.device))
        return clean_images, extra


class ACDiTSeisDimReconNeRFSampler(Sampler):
    """Reconstruct target coordinates using pre-aligned reference files."""

    def setup_dataset(self):
        """Return target coordinates and open reference pairs under the target filenames.

        input_dim_dir defines targets; ref_dir1/2 and ref_dim_dir1/2 contain
        their aligned references. No target signal data is read.
        """
        self.reference_datasets = [
            PairedPatchDataset(*paths) for paths in reference_directories(self.args)
        ]
        return PatchDataset(self.args.input_dim_dir)

    def load_input_batch(self, input_array, input_file, batch_start, batch_end):
        """Load target coordinates and the same patch slice from each reference file.

        Args:
            input_array: Memory-mapped target coordinate array [P, C_dim, H, W].
            input_file: Path identifying the common target/reference filename.
            batch_start: First patch index, inclusive.
            batch_end: Last patch index, exclusive.

        Returns:
            Numpy target coordinates and reference tensor pairs with shapes
            [B, 1, H, W] and [B, C_dim, H, W].
        """
        references = []
        for paired in self.reference_datasets:
            tensors = []
            for dataset in (paired.dataset0, paired.dataset1):
                array = dataset._load_file(dataset.data_path / Path(input_file).name)
                patches = torch.from_numpy(np.array(array[batch_start:batch_end], copy=True)).float()
                if patches.ndim == 3:
                    patches = patches.unsqueeze(1)
                tensors.append(patches)
            references.append(tensors)
        return input_array[batch_start:batch_end], references

    def setup_model(self):
        """Return the ACDiT checkpoint wrapper, optionally applying saved EMA weights."""
        return ACDiT2DWrapper.from_pretrained(
            save_directory=self.args.ckpt, device=self.device, use_ema=self.args.use_ema,
        )

    def preprocess_batch(self, batch):
        """Prepare initial noise, aligned references and fixed reference noise for an ODE run.

        Args:
            batch: Numpy coordinates [B, C_dim, H, W] or [B, H, W], and
                reference signal/coordinate tensor pairs from load_input_batch.

        Returns:
            Noise [B, 1, H, W] and conditioning with encoded x_coord and optional
            r/r_coord/r_noise lists. Clean references and their independent noises
            stay fixed; the wrapper computes r_t at each ODE evaluation's flow time.
        """
        coordinates, reference_pairs = batch
        coordinates = np.array(coordinates, copy=True)
        if coordinates.ndim == 3:
            coordinates = coordinates[:, np.newaxis, :, :]
        coordinates = torch.from_numpy(coordinates).float().to(self.device, non_blocking=True)
        noise = torch.randn(
            coordinates.shape[0], 1, coordinates.shape[-2], coordinates.shape[-1],
            device=self.device, dtype=coordinates.dtype,
        )
        extra = {"x_coord": encode_nerf_conditioning(coordinates, self.args)}
        extra.update(prepare_reference_conditioning(reference_pairs, self.args, self.device))
        return noise, extra


def build_parser():
    """Return the ACDiT CLI with up to two aligned reference directory pairs."""
    parser = argparse.ArgumentParser(
        description="Train or sample ACDiT with NeRF coordinates and aligned noised references.",
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 ACDiTSeisDimReconNeRF.py "
            "--input_dir ./dataset/train --input_dim_dir ./dataset/train_dim "
            "--ref_dir1 ./dataset/train_ref --ref_dim_dir1 ./dataset/train_ref_dim "
            "--ref_dir2 ./dataset/train_ref2 --ref_dim_dir2 ./dataset/train_ref2_dim "
            "--output_dir ./output_acdit --model_arch T --device cuda\n\n"
            "  Sample:\n"
            "  python ACDiTSeisDimReconNeRF.py sample "
            "--ckpt ./output_acdit/run/checkpoint_epoch_01000 "
            "--input_dim_dir ./dataset/valid_dim "
            "--ref_dir1 ./dataset/valid_ref --ref_dim_dir1 ./dataset/valid_ref_dim "
            "--ref_dir2 ./dataset/valid_ref2 --ref_dim_dir2 ./dataset/valid_ref2_dim "
            "--output_dir ./recon_acdit --device cuda\n\n"
            "Omit reference directories to disable cross-attention. Each supplied pair "
            "must match target filenames, per-file patch counts and patch ordering. "
            "All coordinates use the same NeRF settings. References use independent "
            "Gaussian noises at the target flow time. Reference noises stay fixed "
            "within each ODE run."
        ),
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", choices=["train", "sample"], default="train")
    parser.add_argument("--input_dir", default="./dataset/train")
    parser.add_argument("--input_dim_dir", default="./dataset/train_dim")
    for slot in (1, 2):
        parser.add_argument(
            f"--ref_dir{slot}", default=None,
            help=f"Aligned reference {slot} signals under target filenames; omit to disable this slot.",
        )
        parser.add_argument(
            f"--ref_dim_dir{slot}", default=None,
            help=f"Coordinates paired with --ref_dir{slot} in the same file/patch ordering.",
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
