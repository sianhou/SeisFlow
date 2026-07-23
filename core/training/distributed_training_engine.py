import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.dataset import PairedPatchDataset, PatchDataset
from core.logging.logger import build_dist_logger
from core.training.amp_scaler import AMPGradScaler
from core.training.model_utils import count_model_parameters
from core.training.seed import set_random_seed
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from models.wrapper import build_dit_transformer_2d_wrapper, DiTTransformer2DWrapper
from training import distributed_mode


class DistributedTrainingEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.validate_args()
        self.device = None
        self.logger = None
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.flow_path = None
        self.start_epoch = 0

        self.setup_runtime(log_node_info=getattr(self.args, "mode", "train") != "valid")
        if getattr(self.args, "mode", "train") == "valid":
            return
        self.setup_dataset()
        self.setup_model()
        self.wrap_ddp()
        self.setup_optimizer()
        self.setup_scaler()
        self.load_checkpoint()
        self.flow_path = CondOTProbPath()

    def setup_runtime(self, log_node_info=True):
        distributed_mode.init_distributed_mode(self.args)
        self.device = torch.device(self.args.device)
        set_random_seed(self.args.seed + distributed_mode.get_rank())
        self.logger = build_dist_logger(self.args, log_node_info=log_node_info)

    def validate_args(self):
        if getattr(self.args, "mode", "train") == "valid":
            return self.validate_valid_args()
        return self.validate_train_args()

    def validate_train_args(self):
        if not Path(self.args.train_data_dir).is_dir():
            raise FileNotFoundError(f"--train_data_dir must be a directory, got {self.args.train_data_dir}.")
        if not Path(self.args.train_data_dim_dir).is_dir():
            raise FileNotFoundError(f"--train_data_dim_dir must be a directory, got {self.args.train_data_dim_dir}.")
        if self.args.ckpt is not None and not Path(self.args.ckpt).is_dir():
            raise FileNotFoundError(f"--ckpt must be a checkpoint directory, got {self.args.ckpt}.")
        if self.args.input_size <= 0:
            raise ValueError("--input_size must be positive.")
        if self.args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        if self.args.grad_accum_steps <= 0:
            raise ValueError("--grad_accum_steps must be positive.")
        if self.args.num_epochs <= 0:
            raise ValueError("--num_epochs must be positive.")
        if self.args.learning_rate <= 0:
            raise ValueError("--learning_rate must be positive.")
        if self.args.num_workers < 0:
            raise ValueError("--num_workers must be non-negative.")
        if self.args.save_every_epochs <= 0:
            raise ValueError("--save_every_epochs must be positive.")
        if not 0.0 <= self.args.adam_beta1 < 1.0:
            raise ValueError("--adam_beta1 must be in [0, 1).")
        if not 0.0 <= self.args.adam_beta2 < 1.0:
            raise ValueError("--adam_beta2 must be in [0, 1).")

    def validate_valid_args(self):
        if self.args.ckpt is None:
            raise ValueError("--ckpt is required in valid mode.")
        if not Path(self.args.ckpt).is_dir():
            raise FileNotFoundError(f"--ckpt must be a checkpoint directory, got {self.args.ckpt}.")
        if not Path(self.args.train_data_dim_dir).is_dir():
            raise FileNotFoundError(f"--train_data_dim_dir must be a directory, got {self.args.train_data_dim_dir}.")
        if self.args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        if self.args.num_workers < 0:
            raise ValueError("--num_workers must be non-negative.")
        if self.args.solver_step_size <= 0:
            raise ValueError("--solver_step_size must be positive.")
        if self.args.clip_recon is not None and self.args.clip_recon[0] >= self.args.clip_recon[1]:
            raise ValueError("--clip_recon MIN must be smaller than MAX.")

    @staticmethod
    def validate_dim_model_channels(model, dim_channels):
        expected_model_in_channels = 1 + dim_channels
        model_in_channels = int(model.model.config.in_channels)
        model_out_channels = int(model.model.config.out_channels)
        if model_in_channels != expected_model_in_channels:
            raise ValueError(
                "Checkpoint input channel count does not match image + dim data: "
                f"model in_channels={model_in_channels}, dim_channels={dim_channels}, "
                f"expected={expected_model_in_channels}."
            )
        if model_out_channels != 1:
            raise ValueError(
                "Checkpoint output channel count does not match single-channel image patches: "
                f"model out_channels={model_out_channels}."
            )

    @staticmethod
    def split_patch_files_for_rank(dataset):
        rank = distributed_mode.get_rank()
        world_size = distributed_mode.get_world_size()
        return dataset.patch_files[rank::world_size]

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

    def setup_dataset(self):
        self.logger.log_event(
            "dataset_initializing",
            train_data_dir=self.args.train_data_dir,
            train_data_dim_dir=self.args.train_data_dim_dir,
        )

        self.dataset = PairedPatchDataset(self.args.train_data_dir, self.args.train_data_dim_dir)

        self.sampler = torch.utils.data.DistributedSampler(
            self.dataset,
            num_replicas=distributed_mode.get_world_size(),
            rank=distributed_mode.get_rank(),
            shuffle=True,
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=self.sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

        self.dim_channels = self.dataset.dataset1[0].shape[0]
        self.logger.log_event(
            "dataset_initialized",
            dataset_size=len(self.dataset),
            dim_channels=int(self.dim_channels),
            num_batches=len(self.dataloader),
        )

    def setup_model(self):
        self.logger.log_event("model_initializing", model_arch=self.args.model_arch)
        self.model = build_dit_transformer_2d_wrapper(
            model_arch=self.args.model_arch,
            in_channels=1 + self.dim_channels,
            out_channels=1,
            sample_size=self.args.input_size,
            num_embeds_ada_norm=1,
            upcast_attention=self.args.upcast_attention,
            device=self.device,
        )
        self.model_without_ddp = self.model

        total_params, trainable_params, frozen_params = count_model_parameters(self.model_without_ddp)

        effective_batch_size = (
                self.args.batch_size * self.args.grad_accum_steps * distributed_mode.get_world_size()
        )
        self.logger.log_event(
            "model_initialized",
            dataset_size=len(self.dataset),
            num_batches=len(self.dataloader),
            dim_channels=int(self.dim_channels),
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            effective_batch_size=effective_batch_size,
        )

    def wrap_ddp(self):
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.gpu],
                find_unused_parameters=False,
            )
            self.model_without_ddp = self.model.module

    def setup_optimizer(self):
        self.logger.log_event("optimizer_initializing")
        self.optimizer = torch.optim.AdamW(
            self.model_without_ddp.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )

        if self.args.lr_schedule == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=self.args.num_epochs,
                start_factor=1.0,
                end_factor=1e-8 / self.args.learning_rate,
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                total_iters=self.args.num_epochs,
                factor=1.0,
            )

        self.logger.log_event(
            "optimizer_initialized",
            optimizer=str(self.optimizer),
            lr_scheduler=str(self.lr_scheduler),
        )

    def setup_scaler(self):
        self.scaler = AMPGradScaler(enabled=self.device.type == "cuda", device=self.device.type)

    def load_checkpoint(self):
        if not self.args.ckpt:
            return

        self.logger.log_event("checkpoint_loading", path=self.args.ckpt)
        loaded_model, self.start_epoch, training_state = (
            DiTTransformer2DWrapper.from_training(
                save_directory=self.args.ckpt,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                scaler=self.scaler,
                device=self.device,
            )
        )
        del training_state
        self.model_without_ddp.model.load_state_dict(loaded_model.model.state_dict())
        self.logger.log_event(
            "checkpoint_loaded",
            path=self.args.ckpt,
            checkpoint_epoch=self.start_epoch,
            start_epoch=self.start_epoch + 1,
        )

    def model_forward(self, x, t, conditioning):
        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])
        return self.model(
            x,
            t,
            extra={"concat_conditioning": conditioning},
        )

    def train_one_epoch(self, epoch):
        gc.collect()
        self.model.train(True)

        running_loss = 0.0
        running_steps = 0
        epoch_loss = 0.0
        epoch_steps = 0
        total_steps = len(self.dataloader)

        for step, batch in enumerate(self.dataloader):
            if step % self.args.grad_accum_steps == 0:
                self.optimizer.zero_grad()
                running_loss = 0.0
                running_steps = 0

            clean_images, conditioning = batch
            clean_images = clean_images.to(self.device, non_blocking=True)
            conditioning = conditioning.to(self.device, non_blocking=True)
            if clean_images.shape[1] != 1:
                raise ValueError(
                    f"train6.py expects single-channel image patches, got shape {tuple(clean_images.shape)}."
                )
            if clean_images.shape[-2:] != (self.args.input_size, self.args.input_size):
                raise ValueError(
                    f"Expected {self.args.input_size}x{self.args.input_size} patches, "
                    f"got shape {tuple(clean_images.shape)}."
                )
            if conditioning.shape[-2:] != (self.args.input_size, self.args.input_size):
                raise ValueError(
                    f"Expected {self.args.input_size}x{self.args.input_size} dimension patches, "
                    f"got shape {tuple(conditioning.shape)}."
                )

            noise = torch.randn_like(clean_images)
            timesteps = torch.rand(clean_images.shape[0], device=self.device)
            flow_sample = self.flow_path.sample(t=timesteps, x_0=noise, x_1=clean_images)
            noisy_images = flow_sample.x_t
            target_velocity = flow_sample.dx_t

            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                predicted_velocity = self.model_forward(noisy_images, timesteps, conditioning)
                loss = F.mse_loss(predicted_velocity, target_velocity)

            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            running_steps += 1
            epoch_loss += loss_value
            epoch_steps += 1

            scaled_loss = loss / self.args.grad_accum_steps
            should_step = (step + 1) % self.args.grad_accum_steps == 0
            step_start_time = time.time()
            clip_grad = self.args.clip_grad if self.args.clip_grad > 0 else None
            grad_norm = self.scaler(
                scaled_loss,
                self.optimizer,
                clip_grad=clip_grad,
                parameters=self.model.parameters(),
                update_grad=should_step,
            )

            learning_rate = self.optimizer.param_groups[0]["lr"]
            if self.logger is not None:
                self.logger.log_event(
                    "batch_trained",
                    epoch=epoch + 1,
                    step=step + 1,
                    total_steps=total_steps,
                    batch_size=int(clean_images.shape[0]),
                    loss=loss_value,
                    running_loss=running_loss / max(running_steps, 1),
                    lr=learning_rate,
                    optimizer_step=int(should_step),
                    grad_norm="" if grad_norm is None else float(grad_norm.detach().cpu()),
                    clip_grad="" if clip_grad is None else clip_grad,
                    step_time_sec=time.time() - step_start_time,
                )

        return epoch_loss / max(epoch_steps, 1)

    def save_checkpoint(self, epoch, checkpoint_dir):
        checkpoint_path = (
                Path(checkpoint_dir)
                / f"checkpoint_epoch_{epoch + 1:05d}"
        )
        self.model_without_ddp.save_training(
            save_directory=checkpoint_path,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
            args=self.args,
            epoch=epoch + 1,
        )
        self.logger.log_event(
            "checkpoint_saved",
            epoch=epoch + 1,
            path=str(checkpoint_path),
        )

    def run_train(self):
        self.start_time = time.time()
        checkpoint_dir = self.logger.run_dir
        self.logger.log_event("training_started")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            if self.args.distributed:
                self.dataloader.sampler.set_epoch(epoch)

            epoch_loss = self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            self.logger.log_event("epoch_finished", epoch=epoch + 1, mean_loss=epoch_loss, )

            if (
                    (epoch + 1) % self.args.save_every_epochs == 0
                    and distributed_mode.get_rank() == 0
            ):
                self.save_checkpoint(epoch, checkpoint_dir)

        total_time = time.time() - self.start_time
        self.logger.log_event(
            "training_finished",
            total_time_sec=total_time,
            run_dir=str(checkpoint_dir),
        )
        self.logger.close()
        self.cleanup()

    def cleanup(self):
        if getattr(self.args, "distributed", False):
            if getattr(self.args, "dist_backend", None) == "nccl":
                distributed_mode.barrier([self.args.gpu])
            else:
                distributed_mode.barrier()
            distributed_mode.destroy()

    def reconstruct_dim_file(
            self,
            input_file,
            input_root,
            output_root,
            solver,
            time_grid,
            file_index=0,
            total_files=1,
    ):
        dim_array = np.load(input_file, mmap_mode="r")
        num_patches = int(dim_array.shape[0])
        output_file = self.get_reconstruction_output_file(input_file, input_root, output_root)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        reconstructed_batches = []
        file_min = None
        file_max = None

        for batch_start in range(0, num_patches, self.args.batch_size):
            batch_end = min(batch_start + self.args.batch_size, num_patches)
            conditioning = self.make_conditioning_batch(dim_array, batch_start, batch_end, self.device)
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
            sampled = solver.sample(
                time_grid=time_grid,
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
            if self.logger is not None:
                self.logger.log_event(
                    "batch_reconstructed",
                    file=file_index + 1,
                    total_files=total_files,
                    file_name=Path(input_file).name,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    batch_size=batch_end - batch_start,
                )

        reconstructed_patches = torch.cat(reconstructed_batches, dim=0)
        reconstructed_array = self.restore_reconstruction_file_shape(reconstructed_patches, dim_array)
        np.save(output_file, reconstructed_array)

        return {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "num_patches": num_patches,
            "output_shape": list(reconstructed_array.shape),
            "output_min": file_min,
            "output_max": file_max,
        }

    def run_valid(self):
        seed = self.args.seed + distributed_mode.get_rank()
        np.random.seed(seed)

        output_dir = self.logger.run_dir
        self.logger.log_event("checkpoint_loading", path=self.args.ckpt)
        self.model, checkpoint_epoch, training_state = DiTTransformer2DWrapper.from_training(
            save_directory=self.args.ckpt,
            device=self.device,
        )
        del training_state
        self.model_without_ddp = self.model
        self.model.eval()
        self.logger.log_event(
            "checkpoint_loaded",
            path=self.args.ckpt,
            checkpoint_epoch=checkpoint_epoch,
        )

        self.logger.log_event("valid_dataset_initializing", train_data_dim_dir=self.args.train_data_dim_dir)
        dataset = PatchDataset(self.args.train_data_dim_dir)
        dim_channels = dataset[0].shape[0]
        self.validate_dim_model_channels(self.model, dim_channels)
        rank_files = self.split_patch_files_for_rank(dataset)
        self.logger.log_event(
            "valid_dataset_initialized",
            dataset_size=len(dataset),
            dim_channels=int(dim_channels),
            total_files=len(dataset.patch_files),
            rank_files=len(rank_files),
            rank=distributed_mode.get_rank(),
            world_size=distributed_mode.get_world_size(),
        )

        def velocity_model(x, t, cfg_scale=None, label=None, concat_conditioning=None):
            del cfg_scale, label
            with torch.inference_mode():
                with torch.amp.autocast(device_type=x.device.type, enabled=x.device.type == "cuda"):
                    result = self.model_forward(x, t, concat_conditioning["concat_conditioning"])
            return result.to(dtype=torch.float32)

        solver = ODESolver(velocity_model=velocity_model)
        time_grid = torch.tensor([0.0, 1.0], device=self.device)
        file_summaries = []

        self.logger.log_event(
            "validation_started",
            checkpoint=self.args.ckpt,
            train_data_dim_dir=self.args.train_data_dim_dir,
            output_dir=str(output_dir),
            solver_step_size=self.args.solver_step_size,
            clip_recon="" if self.args.clip_recon is None else list(self.args.clip_recon),
            batch_size=self.args.batch_size,
            rank=distributed_mode.get_rank(),
            world_size=distributed_mode.get_world_size(),
        )

        with torch.inference_mode():
            for file_index, input_file in enumerate(rank_files):
                file_summaries.append(
                    self.reconstruct_dim_file(
                        input_file=input_file,
                        input_root=dataset.data_path,
                        output_root=output_dir,
                        solver=solver,
                        time_grid=time_grid,
                        file_index=file_index,
                        total_files=len(rank_files),
                    )
                )

        total_patches = int(sum(item["num_patches"] for item in file_summaries))
        output_min = min(
            (item["output_min"] for item in file_summaries if item["output_min"] is not None),
            default=None,
        )
        output_max = max(
            (item["output_max"] for item in file_summaries if item["output_max"] is not None),
            default=None,
        )

        self.logger.log_event(
            "validation_finished",
            output_dir=str(output_dir),
            num_output_files=len(file_summaries),
            num_patches=total_patches,
            output_min="" if output_min is None else output_min,
            output_max="" if output_max is None else output_max,
            rank=distributed_mode.get_rank(),
        )
        self.logger.close()
        self.cleanup()

    def run(self):
        if getattr(self.args, "mode", "train") == "valid":
            return self.run_valid()
        return self.run_train()
