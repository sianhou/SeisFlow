import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.dataset import PairedPatchDataset, PatchDataset
from core.training import DistributedInference, DistributedTrainer
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from models.dinov2 import DINOv2
from models.nerf import get_nerf_conditioning_channels, encode_nerf_conditioning
from models.wrapper import (
    Pixel_DiT_2D_CONFIGS,
    PixelDiT2DWrapper,
    build_pixeldit_2d_wrapper,
    VelocityModel,
)


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


class PixelDiTSeisDimNeRFReconTrainer(DistributedTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.flow_path = CondOTProbPath()
        self.dino = None
        self.repa_projection = None

    @property
    def repa_enabled(self):
        return self.args.repa_lambda > 0.0

    def build_training_dataset(self):
        return PairedPatchDataset(
            self.args.input_dir,
            self.args.input_dim_dir,
        )

    def build_model(self):
        raw_dim_channels = int(self.dataset.dataset1[0].shape[0])
        nerf_dim_channels = get_nerf_conditioning_channels(raw_dim_channels, self.args)
        model = build_pixeldit_2d_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1 + nerf_dim_channels,
            out_channels=1,
            num_classes=1,
            device=self.device,
        )

        if self.repa_enabled:
            hidden_size = int(model.model.config.hidden_size)
            self.dino = DINOv2(
                model_name=self.args.dino_model_name,
                hub_dir=self.args.dino_hub_dir,
            ).to(self.device)
            self.dino.eval()
            for parameter in self.dino.parameters():
                parameter.requires_grad_(False)

            dino_hidden_size = int(self.dino.encoder.embed_dim)
            self.repa_projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, dino_hidden_size),
            ).to(self.device)
            # Register the projection on the wrapper so the base optimizer sees it.
            model.repa_projection = self.repa_projection

        return model

    def validate_batch(self, clean_images, conditioning):
        if clean_images.shape[1] != 1:
            raise ValueError(
                "PixelDiTSeisDimNeRFRecon expects single-channel image patches, "
                f"got shape {tuple(clean_images.shape)}."
            )

    def train_one_epoch(self, epoch):
        gc.collect()
        self.model.train(True)
        if self.repa_enabled:
            self.repa_projection.train(True)

        running_loss = 0.0
        running_steps = 0
        epoch_loss = 0.0
        epoch_steps = 0
        total_steps = len(self.dataloader)
        accum_steps = 0

        for step, batch in enumerate(self.dataloader):
            if step % self.args.grad_accum_steps == 0:
                self.optimizer.zero_grad()
                running_loss = 0.0
                running_steps = 0
                accum_steps = min(self.args.grad_accum_steps, total_steps - step)

            clean_images, conditioning = batch
            clean_images = clean_images.to(self.device, non_blocking=True)
            conditioning = conditioning.to(self.device, non_blocking=True)
            self.validate_batch(clean_images, conditioning)
            conditioning = encode_nerf_conditioning(conditioning, self.args)

            noise = torch.randn_like(clean_images)
            timesteps = torch.rand(clean_images.shape[0], device=self.device)
            flow_sample = self.flow_path.sample(t=timesteps, x_0=noise, x_1=clean_images)
            noisy_images = flow_sample.x_t
            target_velocity = flow_sample.dx_t

            repa_loss = torch.zeros((), device=self.device)
            feature_buffer = []
            hook_handle = None
            if self.repa_enabled:
                patch_blocks = self.model_without_ddp.model.patch_blocks
                align_index = self.args.repa_align_layer - 1
                hook_handle = patch_blocks[align_index].register_forward_hook(
                    lambda module, inputs, output: feature_buffer.append(output)
                )

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                predicted_velocity = self.model(
                    noisy_images,
                    timesteps,
                    extra={"concat_conditioning": conditioning},
                )
                fm_loss = F.mse_loss(predicted_velocity, target_velocity)

            if self.repa_enabled:
                if len(feature_buffer) != 1:
                    raise RuntimeError(
                        "REPA hook did not capture exactly one patch feature."
                    )
                src_feature = self.repa_projection(feature_buffer[0]).float()
                dino_input = (clean_images + 1.0) / 2.0
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
                loss = fm_loss + self.args.repa_lambda * repa_loss
            else:
                loss = fm_loss

            if hook_handle is not None:
                hook_handle.remove()

            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            running_steps += 1
            epoch_loss += loss_value
            epoch_steps += 1

            scaled_loss = loss / accum_steps
            should_step = (
                    (step + 1) % self.args.grad_accum_steps == 0
                    or step + 1 == total_steps
            )
            step_start_time = time.time()
            clip_grad = self.args.clip_grad if self.args.clip_grad > 0 else None
            grad_norm = self.scaler(
                scaled_loss,
                self.optimizer,
                clip_grad=clip_grad,
                parameters=self.model.parameters(),
                update_grad=should_step,
            )

            self.logger.log_event(
                "batch_done",
                epoch=epoch + 1,
                step=step + 1,
                steps=total_steps,
                loss=loss_value,
                fm_loss=float(fm_loss.detach().cpu()),
                repa_loss=float(repa_loss.detach().cpu()),
                repa_lambda=self.args.repa_lambda,
                avg_loss=running_loss / max(running_steps, 1),
                lr=self.optimizer.param_groups[0]["lr"],
                optimizer_step=bool(should_step),
                grad_norm="" if grad_norm is None else float(grad_norm.detach().cpu()),
                time_sec=time.time() - step_start_time,
            )

        return epoch_loss / max(epoch_steps, 1)

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

    def validate_train_args(self):
        args = self.args
        if not Path(args.input_dir).is_dir():
            raise FileNotFoundError(f"--input_dir must be a directory, got {args.input_dir}.")
        if not Path(args.input_dim_dir).is_dir():
            raise FileNotFoundError(
                f"--input_dim_dir must be a directory, got {args.input_dim_dir}."
            )
        if args.ckpt is not None and not Path(args.ckpt).is_dir():
            raise FileNotFoundError(f"--ckpt must be a checkpoint directory, got {args.ckpt}.")
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        if args.grad_accum_steps <= 0:
            raise ValueError("--grad_accum_steps must be positive.")
        if args.nerf_bands < 0:
            raise ValueError("--nerf_bands must be non-negative.")
        if args.num_epochs <= 0:
            raise ValueError("--num_epochs must be positive.")
        if args.learning_rate <= 0:
            raise ValueError("--learning_rate must be positive.")
        if args.num_workers < 0:
            raise ValueError("--num_workers must be non-negative.")
        if args.save_every_epochs <= 0:
            raise ValueError("--save_every_epochs must be positive.")
        if not 0.0 <= args.adam_beta1 < 1.0:
            raise ValueError("--adam_beta1 must be in [0, 1).")
        if not 0.0 <= args.adam_beta2 < 1.0:
            raise ValueError("--adam_beta2 must be in [0, 1).")
        if args.repa_lambda < 0.0:
            raise ValueError("--repa_lambda must be non-negative.")
        if args.repa_align_layer <= 0:
            raise ValueError("--repa_align_layer must be positive.")

    def load_checkpoint(self):
        super().load_checkpoint()
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

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)
        if not self.is_main_process or not self.repa_enabled:
            return
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch:05d}"
        torch.save(
            self.repa_projection.state_dict(),
            checkpoint_path / "repa_projection.pth",
        )


class PixelDiTSeisDimNeRFReconInference(DistributedInference):
    def build_inference_dataset(self):
        return PatchDataset(self.args.input_dim_dir)

    def get_inference_items(self):
        return self.dataset.patch_files

    def setup_model(self):
        self.model, checkpoint_epoch, _training_state = PixelDiT2DWrapper.from_training(
            save_directory=self.args.ckpt,
            device=self.device,
        )
        self.model.eval()

        raw_dim_channels = int(self.dataset[0].shape[0])
        nerf_dim_channels = self.validate_model_channels(raw_dim_channels)
        self.solver = ODESolver(
            velocity_model=VelocityModel(self.model).to(self.device)
        )
        self.time_grid = torch.tensor([0.0, 1.0], device=self.device)

        self.logger.log_event(
            "checkpoint_loaded",
            path=self.args.ckpt,
            epoch=checkpoint_epoch,
        )
        self.logger.log_event(
            "model_ready",
            dim_channels=nerf_dim_channels,
        )

    def validate_model_channels(self, raw_dim_channels):
        nerf_dim_channels = get_nerf_conditioning_channels(raw_dim_channels, self.args)
        expected_model_in_channels = 1 + nerf_dim_channels
        model_in_channels = int(self.model.model.config.in_channels)
        model_out_channels = int(self.model.model.config.out_channels)
        if model_in_channels != expected_model_in_channels:
            raise ValueError(
                "Checkpoint input channel count does not match image + NeRF dim data: "
                f"model in_channels={model_in_channels}, raw_dim_channels={raw_dim_channels}, "
                f"nerf_dim_channels={nerf_dim_channels}, expected={expected_model_in_channels}."
            )
        if model_out_channels != 1:
            raise ValueError(
                "Checkpoint output channel count does not match single-channel image patches: "
                f"model out_channels={model_out_channels}."
            )
        return nerf_dim_channels

    @staticmethod
    def make_conditioning_batch(array, start, end, device):
        batch = np.array(array[start:end], copy=True)
        if batch.ndim == 3:
            batch = batch[:, np.newaxis, :, :]
        tensor = torch.from_numpy(batch).float()
        return tensor.to(device, non_blocking=True)

    @staticmethod
    def get_reconstruction_output_file(input_file, input_root, output_root):
        relative_path = Path(input_file).relative_to(input_root)
        return Path(output_root) / relative_path

    @staticmethod
    def restore_reconstruction_file_shape(reconstructed, input_array):
        if reconstructed.shape[1] == 1 and input_array.ndim == 3:
            return reconstructed.squeeze(1).numpy()
        return reconstructed.numpy()

    def reconstruct_dim_file(self, input_file, file_index=0, total_files=1):
        dim_array = np.load(input_file, mmap_mode="r")
        num_patches = int(dim_array.shape[0])
        output_file = self.get_reconstruction_output_file(
            input_file,
            self.dataset.data_path,
            self.output_dir,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        reconstructed_batches = []
        file_min = None
        file_max = None

        for batch_start in range(0, num_patches, self.args.batch_size):
            batch_end = min(batch_start + self.args.batch_size, num_patches)
            conditioning = self.make_conditioning_batch(
                dim_array,
                batch_start,
                batch_end,
                self.device,
            )
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
            sampled = self.solver.sample(
                time_grid=self.time_grid,
                x_init=noise,
                return_intermediates=False,
                step_size=self.args.solver_step_size,
                cfg_scale=0.0,
                label=None,
                concat_conditioning={"concat_conditioning": conditioning},
            )
            reconstructed = sampled.detach().float().cpu()
            if self.args.clip_recon is not None:
                reconstructed = reconstructed.clamp(
                    min=float(self.args.clip_recon[0]),
                    max=float(self.args.clip_recon[1]),
                )
            reconstructed_batches.append(reconstructed)
            batch_min = float(reconstructed.min())
            batch_max = float(reconstructed.max())
            file_min = batch_min if file_min is None else min(file_min, batch_min)
            file_max = batch_max if file_max is None else max(file_max, batch_max)
            self.logger.log_event(
                "batch_done",
                file=file_index + 1,
                files=total_files,
                name=Path(input_file).name,
                batch=batch_start // self.args.batch_size + 1,
                batch_size=batch_end - batch_start,
            )

        reconstructed_patches = torch.cat(reconstructed_batches, dim=0)
        reconstructed_array = self.restore_reconstruction_file_shape(
            reconstructed_patches,
            dim_array,
        )
        np.save(output_file, reconstructed_array)

        return {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "num_patches": num_patches,
            "output_shape": list(reconstructed_array.shape),
            "output_min": file_min,
            "output_max": file_max,
        }

    def infer_one_epoch(self):
        file_summaries = []
        with torch.inference_mode():
            for file_index, input_file in enumerate(self.rank_items):
                file_summaries.append(
                    self.reconstruct_dim_file(
                        input_file=input_file,
                        file_index=file_index,
                        total_files=len(self.rank_items),
                    )
                )
        return file_summaries

    def summarize_inference(self, results):
        total_patches = int(sum(item["num_patches"] for item in results))
        output_min = min(
            (item["output_min"] for item in results if item["output_min"] is not None),
            default=None,
        )
        output_max = max(
            (item["output_max"] for item in results if item["output_max"] is not None),
            default=None,
        )
        self.logger.log_event(
            "validation_summary",
            output_dir=str(self.output_dir),
            files=len(results),
            patches=total_patches,
            min="" if output_min is None else output_min,
            max="" if output_max is None else output_max,
        )

    def validate_args(self):
        args = self.args
        if args.ckpt is None:
            raise ValueError("--ckpt is required in valid mode.")
        if not Path(args.ckpt).is_dir():
            raise FileNotFoundError(f"--ckpt must be a checkpoint directory, got {args.ckpt}.")
        if not Path(args.input_dim_dir).is_dir():
            raise FileNotFoundError(
                f"--input_dim_dir must be a directory, got {args.input_dim_dir}."
            )
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        if args.nerf_bands < 0:
            raise ValueError("--nerf_bands must be non-negative.")
        if args.num_workers < 0:
            raise ValueError("--num_workers must be non-negative.")
        if args.solver_step_size <= 0:
            raise ValueError("--solver_step_size must be positive.")
        if args.clip_recon is not None and args.clip_recon[0] >= args.clip_recon[1]:
            raise ValueError("--clip_recon MIN must be smaller than MAX.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train or validate a distributed PixelDiT flow-matching model "
            "from noise to seismic patches with NeRF-encoded dimension coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 DistSeisDimReconNerf.py "
            "--input_dir ./dataset256/train "
            "--input_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_dim_recon "
            "--model_arch T --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Valid:\n"
            "  torchrun --nproc_per_node=4 DistSeisDimReconNerf.py valid "
            "--ckpt ./output_dim_recon/run/checkpoint_epoch_01000 "
            "--input_dim_dir ./dataset256/valid_dim "
            "--output_dir ./seisdimrecon_output "
            "--batch_size 32 --solver_step_size 0.05 --device cuda"
        ),
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "valid"],
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
    parser.add_argument("--upcast_attention", action="store_true")
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
    if args.mode == "valid":
        return PixelDiTSeisDimNeRFReconInference(args).run()
    return PixelDiTSeisDimNeRFReconTrainer(args).run()


if __name__ == "__main__":
    run(build_parser().parse_args())
