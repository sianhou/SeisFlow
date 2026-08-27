import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.dataset import PairedPatchDataset, PatchDataset
from core.training import DistributedInference, DistributedTrainer
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
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


class DiTSeisDimReconExternalNoiseTrainer(DistributedTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.flow_path = CondOTProbPath()

    def build_training_dataset(self):
        return PairedPatchDataset(
            self.args.train_data_dir,
            self.args.train_data_dim_dir,
        )

    def build_model(self):
        dim_channels = int(self.dataset.dataset1[0].shape[0])
        return build_dit_transformer_2d_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1 + dim_channels,
            out_channels=1,
            sample_size=self.args.input_size,
            num_embeds_ada_norm=1,
            upcast_attention=self.args.upcast_attention,
            device=self.device,
        )

    def train_one_epoch(self, epoch):
        gc.collect()
        self.model.train(True)

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

            noise = torch.randn_like(clean_images)
            timesteps = torch.rand(clean_images.shape[0], device=self.device)
            flow_sample = self.flow_path.sample(t=timesteps, x_0=noise, x_1=clean_images)
            noisy_images = flow_sample.x_t
            target_velocity = flow_sample.dx_t

            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                predicted_velocity = self.model(
                    noisy_images,
                    timesteps,
                    extra={"concat_conditioning": conditioning},
                )
                loss = F.mse_loss(predicted_velocity, target_velocity)

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
            if should_step:
                self.update_ema()

            self.logger.log_event(
                "batch_done",
                epoch=epoch + 1,
                step=step + 1,
                steps=total_steps,
                loss=loss_value,
                avg_loss=running_loss / max(running_steps, 1),
                lr=self.optimizer.param_groups[0]["lr"],
                optimizer_step=bool(should_step),
                grad_norm="" if grad_norm is None else float(grad_norm.detach().cpu()),
                time_sec=time.time() - step_start_time,
            )

        return epoch_loss / max(epoch_steps, 1)


class DimVelocityModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(
        self,
        x,
        t,
        cfg_scale,
        label,
        concat_conditioning,
    ):
        del cfg_scale, label

        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])

        with torch.inference_mode():
            with torch.amp.autocast(device_type=x.device.type, enabled=x.device.type == "cuda"):
                result = self.model(x, t, extra=concat_conditioning)
        return result.to(dtype=torch.float32)


class DiTSeisDimReconExternalNoiseInference(DistributedInference):
    def build_inference_dataset(self):
        self.noise_data_path = (
            None if self.args.noise_data_dir is None else Path(self.args.noise_data_dir)
        )
        return PatchDataset(self.args.train_data_dim_dir)

    def get_inference_items(self):
        return self.dataset.patch_files

    def setup_model(self):
        self.model, checkpoint_epoch, _training_state = DiTTransformer2DWrapper.from_pretrained(
            save_directory=self.args.ckpt,
            device=self.device,
            return_training_state=True,
            use_ema=self.args.use_ema,
        )
        self.model.eval()

        dim_channels = int(self.dataset[0].shape[0])
        self.solver = ODESolver(velocity_model=DimVelocityModel(self.model).to(self.device))
        self.time_grid = torch.tensor([0.0, 1.0], device=self.device)

        self.logger.log_event(
            "checkpoint_loaded",
            path=self.args.ckpt,
            epoch=checkpoint_epoch,
        )
        self.logger.log_event(
            "model_ready",
            dim_channels=dim_channels,
        )

    @staticmethod
    def make_batch(array, start, end, device):
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
        noise_array = None
        if self.noise_data_path is not None:
            noise_file = self.noise_data_path / Path(input_file).name
            noise_array = np.load(noise_file, mmap_mode="r")
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
            conditioning = self.make_batch(
                dim_array,
                batch_start,
                batch_end,
                self.device,
            )
            if noise_array is None:
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
                noise = self.make_batch(
                    noise_array,
                    batch_start,
                    batch_end,
                    self.device,
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
            noise_source=(
                "external" if self.noise_data_path is not None else "random"
            ),
            files=len(results),
            patches=total_patches,
            min="" if output_min is None else output_min,
            max="" if output_max is None else output_max,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train or validate DiTSeisDimReconExternalNoise, a distributed conditional "
            "flow-matching model "
            "from noise to seismic patches with dimension coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  Train:\n"
            "  torchrun --nproc_per_node=4 DiTSeisDimReconExternalNoise.py "
            "--train_data_dir ./dataset256/train "
            "--train_data_dim_dir ./dataset256/train_dim "
            "--output_dir ./output_dim_recon "
            "--model_arch DiT_T_4 --input_size 256 --batch_size 32 --num_epochs 1000 --device cuda\n\n"
            "  Valid:\n"
            "  torchrun --nproc_per_node=4 DiTSeisDimReconExternalNoise.py valid "
            "--ckpt ./output_dim_recon/run/checkpoint_epoch_01000 "
            "--train_data_dim_dir ./dataset256/valid_dim "
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
    parser.add_argument("--train_data_dir", default="./dataset/train")
    parser.add_argument("--train_data_dim_dir", default="./dataset/train_dim")
    parser.add_argument(
        "--noise_data_dir",
        default=None,
        help=(
            "Optional directory of external single-channel noise patch NPY files "
            "for valid mode, matched to dimension patches by file name. "
            "When omitted, noise is generated with torch.randn."
        ),
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
        help="Maintain EMA weights and use them for inference when loading a checkpoint.",
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


def run_train(args):
    args.mode = "train"
    trainer = DiTSeisDimReconExternalNoiseTrainer(args)
    return trainer.run()


def run_valid(args):
    args.mode = "valid"
    inference = DiTSeisDimReconExternalNoiseInference(args)
    return inference.run()


def run(args):
    if args.mode == "valid":
        return run_valid(args)
    return run_train(args)


if __name__ == "__main__":
    run(build_parser().parse_args())
