import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from core.logging.logger import build_dist_logger
from training import distributed_mode
from .amp_scaler import AMPGradScaler
from .model_utils import count_model_parameters
from .seed import set_random_seed


class DistributedTrainer(ABC):
    """Base class for torch DistributedDataParallel training scripts."""

    def __init__(self, args):
        self.args = args
        self.logger = None
        self.device = None
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.start_epoch = 0
        self.checkpoint_dir = None

    @property
    def is_main_process(self):
        return distributed_mode.get_rank() == 0

    def setup_runtime(self):
        distributed_mode.init_distributed_mode(self.args)
        self.logger = build_dist_logger(self.args, log_node_info=True)
        self.device = torch.device(self.args.device)
        set_random_seed(self.args.seed + distributed_mode.get_rank())

    def setup_dataset(self):
        self.logger.log_event("dataset_loading")

        self.dataset = self.build_training_dataset()

        sampler = torch.utils.data.DistributedSampler(
            self.dataset,
            num_replicas=distributed_mode.get_world_size(),
            rank=distributed_mode.get_rank(),
            shuffle=True,
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )
        
        self.logger.log_event(
            "dataset_ready",
            samples=len(self.dataset),
            batches=len(self.dataloader),
        )

    @abstractmethod
    def build_training_dataset(self):
        """Return the training dataset."""

    @abstractmethod
    def build_model(self):
        """Return the unwrapped model on self.device."""

    def setup_model(self):
        self.model = self.build_model()
        self.model_without_ddp = self.model

        total_params, trainable_params, frozen_params = count_model_parameters(self.model_without_ddp)
        del frozen_params
        self.logger.log_event(
            "model_ready",
            arch=getattr(self.args, "model_arch", ""),
            params=total_params,
            trainable_params=trainable_params,
        )

        if getattr(self.args, "distributed", False):
            ddp_kwargs = {"find_unused_parameters": False}
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.args.gpu]
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                **ddp_kwargs,
            )
            self.model_without_ddp = self.model.module

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

        self.scaler = AMPGradScaler(enabled=self.device.type == "cuda", device=self.device.type)
        effective_batch_size = (
                self.args.batch_size
                * self.args.grad_accum_steps
                * distributed_mode.get_world_size()
        )
        self.logger.log_event(
            "optimizer_ready",
            lr=self.args.learning_rate,
            lr_schedule=self.args.lr_schedule,
            effective_batch_size=effective_batch_size,
        )

    def load_checkpoint(self):
        if not getattr(self.args, "ckpt", None):
            return

        loaded_model, checkpoint_epoch, _training_state = type(self.model_without_ddp).from_pretrained(
            save_directory=self.args.ckpt,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
            device=self.device,
            return_training_state=True,
        )
        self.load_model_state(loaded_model)
        self.start_epoch = int(checkpoint_epoch)
        self.logger.log_event(
            "checkpoint_loaded",
            path=self.args.ckpt,
            epoch=self.start_epoch,
            start_epoch=self.start_epoch + 1,
        )

    def load_model_state(self, loaded_model):
        if hasattr(self.model_without_ddp, "model") and hasattr(loaded_model, "model"):
            self.model_without_ddp.model.load_state_dict(loaded_model.model.state_dict())
            return
        self.model_without_ddp.load_state_dict(loaded_model.state_dict())

    @abstractmethod
    def train_one_epoch(self, epoch):
        """Train one epoch and return the mean loss."""

    def save_checkpoint(self, epoch):
        if not self.is_main_process:
            return

        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch:05d}"
        self.model_without_ddp.save_pretrained(
            save_directory=checkpoint_path,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
            args=self.args,
            epoch=epoch,
        )
        self.logger.log_event("checkpoint_saved", epoch=epoch, path=str(checkpoint_path))

    def train(self):
        self.logger.log_event("train_started", epochs=self.args.num_epochs)
        start_time = time.time()
        self.checkpoint_dir = self.logger.run_dir

        for epoch in range(self.start_epoch, self.args.num_epochs):
            if getattr(self.args, "distributed", False) and hasattr(self.dataloader.sampler, "set_epoch"):
                self.dataloader.sampler.set_epoch(epoch)

            epoch_loss = self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            self.logger.log_event("epoch_done", epoch=epoch + 1, loss=epoch_loss)

            if (epoch + 1) % self.args.save_every_epochs == 0:
                self.save_checkpoint(epoch + 1)

        self.logger.log_event(
            "train_done",
            total_time_sec=time.time() - start_time,
            run_dir=str(self.checkpoint_dir),
        )

    def run(self):
        try:
            self.validate_train_args()
            self.setup_runtime()
            self.setup_dataset()
            self.setup_model()
            self.setup_optimizer()
            self.load_checkpoint()
            self.train()
        finally:
            self.close()

    def close(self):
        if self.logger is not None:
            self.logger.close()
        if getattr(self.args, "distributed", False):
            if self.device is not None and self.device.type == "cuda":
                distributed_mode.barrier([self.args.gpu])
            else:
                distributed_mode.barrier()
            distributed_mode.destroy()

    def validate_train_args(self):
        return None


class DistributedInference(ABC):
    """Base class for distributed validation or inference scripts."""

    def __init__(self, args):
        self.args = args
        self.logger = None
        self.device = None
        self.dataset = None
        self.rank_items = None
        self.model = None
        self.output_dir = None

    @property
    def is_main_process(self):
        return distributed_mode.get_rank() == 0

    def setup_runtime(self):
        distributed_mode.init_distributed_mode(self.args)
        self.logger = build_dist_logger(self.args, log_node_info=True)
        self.device = torch.device(self.args.device)
        set_random_seed(self.args.seed + distributed_mode.get_rank())
        self.output_dir = self.logger.run_dir

    def setup_dataset(self):
        self.logger.log_event("dataset_loading")
        self.dataset = self.build_inference_dataset()
        inference_items = self.get_inference_items()
        self.rank_items = self.split_items_for_rank(inference_items)
        self.logger.log_event(
            "dataset_ready",
            samples=len(self.dataset),
            total_items=len(inference_items),
            rank_items=len(self.rank_items),
        )

    @abstractmethod
    def build_inference_dataset(self):
        """Return the dataset used by validation or inference."""

    def get_inference_items(self):
        return list(range(len(self.dataset)))

    def split_items_for_rank(self, items):
        rank = distributed_mode.get_rank()
        world_size = distributed_mode.get_world_size()
        return list(items)[rank::world_size]

    @abstractmethod
    def setup_model(self):
        """Load and prepare the model for validation or inference."""

    def inference(self):
        self.logger.log_event(
            "validation_started",
            output_dir=str(self.output_dir),
        )
        start_time = time.time()
        results = self.infer_one_epoch()
        self.summarize_inference(results)
        self.logger.log_event(
            "validation_done",
            total_time_sec=time.time() - start_time,
            output_dir=str(self.output_dir),
        )
        return results

    @abstractmethod
    def infer_one_epoch(self):
        """Run one validation or inference pass over this rank's items."""

    def summarize_inference(self, results):
        self.logger.log_event("validation_summary", results=len(results))

    def run(self):
        try:
            self.validate_args()
            self.setup_runtime()
            self.setup_dataset()
            self.setup_model()
            return self.inference()
        finally:
            self.close()

    def close(self):
        if self.logger is not None:
            self.logger.close()
        if getattr(self.args, "distributed", False):
            if self.device is not None and self.device.type == "cuda":
                distributed_mode.barrier([self.args.gpu])
            else:
                distributed_mode.barrier()
            distributed_mode.destroy()

    def validate_args(self):
        return None
