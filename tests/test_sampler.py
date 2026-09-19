import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from core.sampler import DistributedInferenceSampler, Sampler


def collect_rank_indices(size, world_size, shuffle=False, seed=0, epoch=0):
    rank_indices = []
    for rank in range(world_size):
        sampler = DistributedInferenceSampler(
            size,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        sampler.set_epoch(epoch)
        rank_indices.append(list(sampler))
    return rank_indices


def test_distributed_inference_sampler_partitions_without_duplicates():
    rank_indices = collect_rank_indices(size=10, world_size=3)

    assert rank_indices == [
        [0, 3, 6, 9],
        [1, 4, 7],
        [2, 5, 8],
    ]
    flattened = [index for indices in rank_indices for index in indices]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_distributed_inference_sampler_supports_empty_ranks():
    rank_indices = collect_rank_indices(size=2, world_size=4)

    assert rank_indices == [[0], [1], [], []]
    assert [
        len(DistributedInferenceSampler(2, num_replicas=4, rank=rank))
        for rank in range(4)
    ] == [1, 1, 0, 0]


def test_distributed_inference_sampler_shuffle_is_global_and_reproducible():
    first = collect_rank_indices(
        size=17,
        world_size=4,
        shuffle=True,
        seed=123,
        epoch=5,
    )
    repeated = collect_rank_indices(
        size=17,
        world_size=4,
        shuffle=True,
        seed=123,
        epoch=5,
    )
    next_epoch = collect_rank_indices(
        size=17,
        world_size=4,
        shuffle=True,
        seed=123,
        epoch=6,
    )

    assert first == repeated
    assert first != next_epoch
    flattened = [index for indices in first for index in indices]
    assert sorted(flattened) == list(range(17))
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize(
    ("num_replicas", "rank"),
    [(0, 0), (2, -1), (2, 2)],
)
def test_distributed_inference_sampler_rejects_invalid_rank_configuration(
        num_replicas,
        rank,
):
    with pytest.raises(ValueError):
        DistributedInferenceSampler(
            4,
            num_replicas=num_replicas,
            rank=rank,
        )


class ExampleSampler(Sampler):
    def setup_dataset(self):
        return list(range(7))

    def file_list(self):
        return [f"item-{index}" for index in self.dataset]

    def setup_model(self):
        return nn.Linear(1, 1)


class DefaultFileSampler(ExampleSampler):
    def preprocess_batch(self, batch):
        tensor = torch.from_numpy(np.array(batch, copy=True)).float()
        return tensor.unsqueeze(1), None


class OffsetSolver:
    def sample(self, x_init, **kwargs):
        del kwargs
        return x_init + 2.0


def test_sampler_base_assigns_items_using_rank_and_world_size():
    sampler = ExampleSampler(
        SimpleNamespace(
            seed=0,
            sampler_shuffle=False,
        )
    )
    sampler.rank = 1
    sampler.world_size = 3

    sampler._setup_dataset()

    assert sampler.rank_files == ["item-1", "item-4"]
    assert list(sampler.file_sampler) == [1, 4]


def test_sampler_setup_model_owns_device_and_eval_configuration():
    sampler = ExampleSampler(
        SimpleNamespace(
            seed=0,
            sampler_shuffle=False,
        )
    )
    sampler.device = torch.device("cpu")

    sampler._setup_model()

    assert sampler.model.training is False
    assert next(sampler.model.parameters()).device.type == "cpu"


def test_sampler_provides_default_ode_solver_and_time_grid(monkeypatch):
    torchdiffeq = ModuleType("torchdiffeq")
    torchdiffeq.odeint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchdiffeq", torchdiffeq)
    sampler = ExampleSampler(
        SimpleNamespace(
            seed=0,
            sampler_shuffle=False,
        )
    )
    sampler.device = torch.device("cpu")
    sampler._setup_model()

    sampler.setup_sampler()

    assert sampler.solver.velocity_model.model is sampler.model
    assert torch.equal(sampler.time_grid, torch.tensor([0.0, 1.0]))
    assert sampler.time_grid.device.type == "cpu"


def test_sampler_preprocess_batch_matches_trainer_default_contract():
    sampler = ExampleSampler(
        SimpleNamespace(
            seed=0,
            sampler_shuffle=False,
        )
    )
    batch = torch.randn(2, 3)

    processed, extra = sampler.preprocess_batch(batch)

    assert processed is batch
    assert extra is None


def test_sampler_default_epoch_samples_batches_and_saves_files(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    input_file = input_dir / "patches.npy"
    np.save(input_file, np.zeros((3, 2, 2), dtype=np.float32))

    sampler = DefaultFileSampler(
        SimpleNamespace(
            batch_size=2,
            solver_step_size=0.1,
            clip_recon=(-0.5, 0.5),
        )
    )
    sampler.dataset = SimpleNamespace(data_path=input_dir)
    sampler.rank_files = [str(input_file)]
    sampler.output_dir = output_dir
    sampler.solver = OffsetSolver()
    sampler.time_grid = torch.tensor([0.0, 1.0])
    events = []
    sampler.logger = SimpleNamespace(
        log_event=lambda event, **values: events.append((event, values))
    )

    sampler.sample_one_epoch()

    output = np.load(output_dir / "patches.npy")
    assert output.shape == (3, 2, 2)
    assert np.all(output == 0.5)
    assert [event for event, _ in events] == ["batch_done", "batch_done"]
