"""Distributed sampling and inference primitives."""

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler as TorchSampler

from core.logging.logger import build_dist_logger
from core.trainer import Dist
from core.training.model_utils import count_model_parameters
from core.training.seed import set_random_seed


class DistributedInferenceSampler(TorchSampler[int]):
    """Partition inference indices across ranks without padding or duplication.

    Unlike PyTorch's training-oriented ``DistributedSampler``, this sampler
    never repeats indices to make all ranks equally long. That property is
    important for file-producing inference jobs, where duplicate indices can
    make multiple ranks write the same output.
    """

    def __init__(
            self,
            data_source,
            num_replicas=None,
            rank=None,
            shuffle=False,
            seed=0,
    ):
        if isinstance(data_source, int):
            self.dataset_size = int(data_source)
        else:
            self.dataset_size = len(data_source)
        if self.dataset_size < 0:
            raise ValueError("DistributedInferenceSampler size cannot be negative.")

        if num_replicas is None:
            num_replicas = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0
            )

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}."
            )

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        if not self.shuffle:
            return iter(
                range(
                    self.rank,
                    self.dataset_size,
                    self.num_replicas,
                )
            )

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.dataset_size,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        remaining = self.dataset_size - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class Sampler(Dist, ABC):
    """Base class for distributed model sampling and inference jobs."""

    def __init__(self, args):
        super().__init__(args)
        self.logger = None
        self.device = None
        self.dataset = None
        self.files = None
        self.file_sampler = None
        self.rank_files = None
        self.model = None
        self.output_dir = None

    def _setup_logger(self):
        self.logger = build_dist_logger(self.args, log_node_info=True)
        return self.logger

    def _setup_runtime(self):
        self._setup_distributed()
        self.device = torch.device(self.args.device)
        self._setup_logger()
        set_random_seed(self.args.seed + self.rank)
        self.output_dir = self.logger.run_dir

    @abstractmethod
    def setup_dataset(self):
        """Return the dataset used to prepare sampling items."""
        raise NotImplementedError

    def file_list(self):
        return self.dataset.patch_files

    def split_files_for_rank(self, files):
        files = list(files)
        self.file_sampler = DistributedInferenceSampler(
            files,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=getattr(self.args, "sampler_shuffle", False),
            seed=getattr(self.args, "seed", 0),
        )
        return [files[index] for index in self.file_sampler]

    def _setup_dataset(self):
        if self.logger is not None:
            self.logger.log_event("dataset_loading")

        self.dataset = self.setup_dataset()
        self.files = list(self.file_list())
        self.rank_files = self.split_files_for_rank(self.files)

        if self.logger is not None:
            self.logger.log_event(
                "dataset_ready",
                samples=len(self.dataset),
                total_files=len(self.files),
                rank_files=len(self.rank_files),
            )

    def preprocess_batch(self, batch):
        return batch, None

    @abstractmethod
    def setup_model(self):
        """Configure and return the model used for sampling."""
        raise NotImplementedError

    def _setup_model(self):
        self.model = self.setup_model().to(self.device)
        self.model.eval()

        if self.logger is not None:
            total, trainable, _ = count_model_parameters(self.model)
            self.logger.log_event(
                "model_ready",
                params=total,
                trainable_params=trainable,
            )

    def setup_sampler(self):
        """Configure solvers or other state needed after the model is ready."""
        from flow_matching.solver import ODESolver
        from models.wrapper import VelocityModel

        self.solver = ODESolver(
            velocity_model=VelocityModel(self.model).to(self.device)
        )
        self.time_grid = torch.tensor([0.0, 1.0], device=self.device)

    def sample_one_epoch(self):
        """Sample and save every assigned NPY patch file.

        Override this method only when a task does not use the default
        file-based patch reconstruction workflow.
        """
        total_files = len(self.rank_files)

        with torch.inference_mode():
            for file_index, input_file in enumerate(self.rank_files):
                input_array = np.load(input_file, mmap_mode="r")
                num_patches = int(input_array.shape[0])
                if num_patches == 0:
                    raise ValueError(
                        f"Sampling input contains no patches: {input_file}"
                    )

                relative_path = Path(input_file).relative_to(
                    self.dataset.data_path
                )
                output_file = Path(self.output_dir) / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                sampled_batches = []

                for batch_start in range(0, num_patches, self.args.batch_size):
                    batch_end = min(batch_start + self.args.batch_size, num_patches, )
                    x_init, extra = self.preprocess_batch(
                        input_array[batch_start:batch_end]
                    )
                    sampled = self.solver.sample(
                        time_grid=self.time_grid,
                        x_init=x_init,
                        return_intermediates=False,
                        step_size=self.args.solver_step_size,
                        cfg_scale=0.0,
                        label=None,
                        concat_conditioning=extra,
                    )
                    sampled = sampled.detach().float().cpu()

                    clip_recon = getattr(self.args, "clip_recon", None)
                    if clip_recon is not None:
                        sampled = sampled.clamp(
                            min=float(clip_recon[0]),
                            max=float(clip_recon[1]),
                        )

                    sampled_batches.append(sampled)

                    if self.logger is not None:
                        self.logger.log_event(
                            "batch_done",
                            file=file_index + 1,
                            files=total_files,
                            name=Path(input_file).name,
                            batch=batch_start // self.args.batch_size + 1,
                            batch_size=batch_end - batch_start,
                        )

                sampled_array = torch.cat(sampled_batches, dim=0)
                if sampled_array.shape[1] == 1 and input_array.ndim == 3:
                    sampled_array = sampled_array.squeeze(1)
                sampled_array = sampled_array.numpy()
                np.save(output_file, sampled_array)

    def sample(self):
        if self.logger is not None:
            self.logger.log_event(
                "sampling_started",
                output_dir=str(self.output_dir),
            )
        start_time = time.time()
        self.sample_one_epoch()
        if self.logger is not None:
            self.logger.log_event(
                "sampling_done",
                total_time_sec=time.time() - start_time,
                output_dir=str(self.output_dir),
            )

    def run(self):
        try:
            self._setup_runtime()
            self._setup_dataset()
            self._setup_model()
            self.setup_sampler()
            self.sample()
        finally:
            self._close()

    def _close(self):
        if self.logger is not None:
            self.logger.close()
        self._barrier()
        self._destroy_distributed()
