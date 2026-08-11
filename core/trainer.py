"""Independent base trainer for Flow Matching experiments."""

import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.training_utils import EMAModel as EMA
from torch.nn.parallel import DistributedDataParallel

from core.logging.logger import build_dist_logger
from .training.amp_scaler import AMPGradScaler
from .training.model_utils import count_model_parameters


class Dist:
    def __init__(self, args):
        self.args = args
        self.rank = 0
        self.world_size = 1

    def _setup_distributed(self):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.args.distributed = self.world_size > 1

            if torch.cuda.is_available() and self.args.device.startswith("cuda"):
                self.args.gpu = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(self.args.gpu)
                backend = "nccl"
            else:
                backend = "gloo"

            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            self.args.distributed = False

    def _barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _destroy_distributed(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_main_process(self):
        return self.rank == 0


class Trainer(Dist, ABC):
    def __init__(self, args):
        super().__init__(args)
        self.logger = None
        self.device = None
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema = None
        self.start_epoch = 0
        self.checkpoint_dir = None

    def _setup_logger(self):
        self.logger = build_dist_logger(self.args, log_node_info=True)
        return self.logger

    def _setup_seed(self, seed: int = 0, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None and cudnn.is_available():
            cudnn.deterministic = deterministic
            cudnn.benchmark = not deterministic

    def _setup_runtime(self):
        self._setup_distributed()
        self.device = torch.device(self.args.device)
        self._setup_logger()
        self._setup_seed(self.args.seed + self.rank)

    @abstractmethod
    def setup_dataset(self):
        raise NotImplementedError

    def _setup_dataset(self):
        self.dataset = self.setup_dataset()
        sampler = torch.utils.data.DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        ) if getattr(self.args, "distributed", False) else None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.args.num_workers,
            pin_memory=getattr(self.args, "pin_memory", False),
            drop_last=True,
        )

    def preprocess_batch(self, batch):
        return batch, None

    def sample_path(self, x1, mode="linear"):
        if mode != "linear":
            raise ValueError(f"Unsupported path mode: {mode}")

        noise = torch.randn_like(x1)
        t = torch.rand(
            x1.shape[0],
            device=x1.device,
            dtype=x1.dtype,
        )
        t_view = t.view(
            x1.shape[0],
            *([1] * (x1.ndim - 1)),
        )

        x_t = (1.0 - t_view) * noise + t_view * x1
        velocity = x1 - noise

        return {
            "x_t": x_t,
            "t": t,
            "x1": x1,
            "noise": noise,
            "velocity": velocity,
        }

    def compute_auxiliary_loss(self):
        return 0

    def compute_loss(self, prediction, sample, mode="velocity"):

        if mode == "velocity":
            target = sample["velocity"]
        elif mode == "x1":
            target = sample["x1"]
        elif mode == "noise":
            target = sample["noise"]
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")

        loss = F.mse_loss(
            prediction.float(),
            target.float(),
        )

        auxiliary_loss = self.compute_auxiliary_loss()
        total_loss = loss + auxiliary_loss
        return total_loss, loss, auxiliary_loss

    @abstractmethod
    def setup_model(self):
        """Return the unwrapped model. Implement this in the subclass."""
        raise NotImplementedError

    def _setup_model(self):
        self.model = self.setup_model().to(self.device)
        self.model_without_ddp = self.model

        if self.logger is not None:
            total, trainable, _ = count_model_parameters(self.model)
            self.logger.log_event(
                "model_ready",
                params=total,
                trainable_params=trainable,
            )

        if getattr(self.args, "distributed", False):
            kwargs = {"find_unused_parameters": False}
            if self.device.type == "cuda":
                kwargs["device_ids"] = [self.args.gpu]

            self.model = DistributedDataParallel(
                self.model,
                **kwargs,
            )
            self.model_without_ddp = self.model.module

        if getattr(self.args, "use_ema", False):
            ema_model = getattr(
                self.model_without_ddp,
                "model",
                self.model_without_ddp,
            )
            self.ema = EMA(
                ema_model.parameters(),
                decay=self.args.ema_decay,
                update_after_step=self.args.ema_warmup,
                model_cls=type(ema_model),
                model_config=ema_model.config,
            )

    def setup_optimizer(self):
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

        self.scaler = AMPGradScaler(
            enabled=self.device.type == "cuda",
            device=self.device.type,
        )

    def update_ema(self):
        if self.ema is None:
            return
        ema_model = getattr(
            self.model_without_ddp,
            "model",
            self.model_without_ddp,
        )
        self.ema.step(ema_model.parameters())

    def from_pretrained(self):
        checkpoint = getattr(self.args, "ckpt", None)
        if checkpoint is None:
            return

        loaded, epoch, _ = type(self.model_without_ddp).from_pretrained(
            save_directory=checkpoint,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
            device=self.device,
            return_training_state=True,
            ema=self.ema,
        )

        if hasattr(self.model_without_ddp, "model"):
            self.model_without_ddp.model.load_state_dict(
                loaded.model.state_dict()
            )
        else:
            self.model_without_ddp.load_state_dict(loaded.state_dict())

        self.start_epoch = int(epoch)

    def save_pretrained(self, epoch):
        if not self.is_main_process:
            return

        path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch:05d}"
        self.model_without_ddp.save_pretrained(
            save_directory=path,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
            args=self.args,
            epoch=epoch,
            ema=self.ema,
        )

    def train_one_epoch(self, epoch):
        self.model.train()

        epoch_loss = 0.0
        total_steps = len(self.dataloader)
        accum_steps = max(
            int(getattr(self.args, "grad_accum_steps", 1)),
            1,
        )

        for step, batch in enumerate(self.dataloader):
            step_start_time = time.time()

            if step % accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            x, extra = self.preprocess_batch(batch)
            sample = self.sample_path(x)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                prediction = self.model(
                    sample["x_t"],
                    sample["t"],
                    extra=extra,
                )

                total_loss, loss, auxiliary_loss = self.compute_loss(
                    prediction,
                    sample,
                )

            should_step = (
                (step + 1) % accum_steps == 0
                or step + 1 == total_steps
            )
            total_loss_value = total_loss.detach()
            loss_value = loss.detach()
            auxiliary_loss_value = torch.as_tensor(
                auxiliary_loss,
                device=total_loss.device,
            ).detach()
            total_loss = total_loss / min(accum_steps, total_steps - step)
            clip_grad = getattr(self.args, "clip_grad", 0.0)

            grad_norm = self.scaler(
                total_loss,
                self.optimizer,
                clip_grad=clip_grad if clip_grad > 0 else None,
                parameters=self.model.parameters(),
                update_grad=should_step,
            )

            if should_step:
                self.update_ema()

            epoch_loss += float(total_loss_value.cpu())

            if self.logger is not None:
                self.logger.log_event(
                    "batch_done",
                    epoch=epoch + 1,
                    step=step + 1,
                    steps=total_steps,
                    total_loss=float(total_loss_value.cpu()),
                    loss=float(loss_value.cpu()),
                    auxiliary_loss=float(auxiliary_loss_value.cpu()),
                    lr=self.optimizer.param_groups[0]["lr"],
                    optimizer_step=bool(should_step),
                    grad_norm=(
                        ""
                        if grad_norm is None
                        else float(grad_norm.detach().cpu())
                    ),
                    time_sec=time.time() - step_start_time,
                )

        return epoch_loss / max(total_steps, 1)

    def train(self):
        self.checkpoint_dir = self.logger.run_dir

        for epoch in range(self.start_epoch, self.args.num_epochs):
            loss = self.train_one_epoch(epoch)
            self.lr_scheduler.step()

            if self.logger is not None:
                self.logger.log_event(
                    "epoch_done",
                    epoch=epoch + 1,
                    loss=loss,
                )

            if (epoch + 1) % self.args.save_every_epochs == 0:
                self.save_pretrained(epoch + 1)

    def run(self):
        try:
            self._setup_runtime()
            self._setup_dataset()
            self._setup_model()
            self.setup_optimizer()
            self.from_pretrained()
            self.train()
        finally:
            self._close()

    def _close(self):
        if self.logger is not None:
            self.logger.close()
        self._barrier()
        self._destroy_distributed()
