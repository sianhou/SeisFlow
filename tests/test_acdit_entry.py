"""Verify paired random references and ACDiT training/reconstruction integration."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

import ACDiTSeisDimReconNeRF as entry
from core.dataset import PairedPatchDataset
from models.acdit2d import ACDiT2DWrapper
from models.nerf import encode_nerf_conditioning
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS


@pytest.fixture
def paired_dirs(tmp_path):
    """Write five paired patches across two files and return their directories.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        Signal and coordinate directory paths. Signals are [P, 1, 4, 6];
        coordinates [P, 5, 4, 6] retain patch identity in their first channel.
    """
    signal_dir, coord_dir = tmp_path / 'data', tmp_path / 'coords'
    signal_dir.mkdir()
    coord_dir.mkdir()
    data = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None, None] / 5,
                           (5, 1, 4, 6)).copy()
    coords = np.concatenate([data + i / 10 for i in range(5)], axis=1)
    for name, indices in [('a.npy', slice(0, 3)), ('b.npy', slice(3, 5))]:
        np.save(signal_dir / name, data[indices])
        np.save(coord_dir / name, coords[indices])
    return signal_dir, coord_dir


def make_args(paired_dirs, use_ref=0):
    """Return a CPU CLI namespace suitable for small integration tests.

    Args:
        paired_dirs: Signal and coordinate directory paths.
        use_ref: Number of random references per target; zero disables them.

    Returns:
        Parsed training arguments with a small batch and one training epoch.
    """
    return entry.build_parser().parse_args([
        '--input_dir', str(paired_dirs[0]), '--input_dim_dir', str(paired_dirs[1]),
        '--device', 'cpu', '--model_arch', 'Nano', '--patch_size', '2',
        '--batch_size', '2', '--num_workers', '0', '--num_epochs', '1',
        '--save_every_epochs', '1', '--use_ref', str(use_ref), '--nerf_bands', '1',
        '--solver_step_size', '0.5',
    ])


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('include_input', [False, True])
def test_random_references_preserve_pairs(paired_dirs, count, include_input):
    """Random references preserve signal/coordinate alignment and obey seeded RNG.

    Args:
        paired_dirs: Temporary paired NPY directories.
        count: Reference count per target.
        include_input: Include raw coordinates in the NeRF encoding.
    """
    args = make_args(paired_dirs, count)
    args.nerf_include_input = include_input
    dataset = PairedPatchDataset(*paired_dirs)
    torch.manual_seed(19)
    indices = torch.randint(len(dataset), (count, 2))
    torch.manual_seed(19)
    extra = entry.sample_reference_conditioning(dataset, 2, args, 'cpu')
    assert len(extra['r']) == len(extra['r_coord']) == count
    for index, row in enumerate(indices.tolist()):
        expected_data = torch.stack([dataset[item][0] for item in row])
        expected_coord = torch.stack([dataset[item][1] for item in row])
        torch.testing.assert_close(extra['r'][index], expected_data)
        torch.testing.assert_close(extra['r_coord'][index], encode_nerf_conditioning(expected_coord, args))


def test_reference_pool_includes_self_and_repeated_draws(paired_dirs):
    """A one-item reference pool can supply itself repeatedly without exclusion.

    Args:
        paired_dirs: Temporary paired NPY directories.
    """
    dataset = Subset(PairedPatchDataset(*paired_dirs), [0])
    args = make_args(paired_dirs, use_ref=3)
    extra = entry.sample_reference_conditioning(dataset, 2, args, 'cpu')
    for reference in extra['r']:
        torch.testing.assert_close(reference, dataset[0][0].unsqueeze(0).expand(2, -1, -1, -1))


@pytest.mark.parametrize('count', [0, 1, 3])
def test_trainer_preprocess_and_model(paired_dirs, monkeypatch, count):
    """Train batches keep clean references separate from target noise and coordinates.

    Args:
        paired_dirs: Temporary paired NPY directories.
        monkeypatch: Pytest fixture supplying a small architecture preset.
        count: Random reference count, including the disabled path.
    """
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, 'Nano',
                        dict(hidden_size=16, num_groups=2, depth=2))
    trainer = entry.ACDiTSeisDimReconNeRFTrainer(make_args(paired_dirs, count))
    trainer.device = torch.device('cpu')
    trainer.dataset = trainer.setup_dataset()
    batch = next(iter(DataLoader(trainer.dataset, batch_size=2)))
    clean, extra = trainer.preprocess_batch(batch)
    torch.testing.assert_close(clean, batch[0])
    torch.testing.assert_close(extra['x_coord'], encode_nerf_conditioning(batch[1], trainer.args))
    if count:
        assert len(extra['r']) == len(extra['r_coord']) == count
        for signal, coords in zip(extra['r'], extra['r_coord']):
            torch.testing.assert_close(signal[:, 0], coords[:, 0])
    else:
        assert set(extra) == {'x_coord'}
    model = trainer.setup_model()
    assert model.model.use_cross_attention == bool(count)
    assert model.model.in_coords_channels == 15
    sample = trainer.sample_path(clean)
    prediction = model(sample['x_t'], sample['t'], extra)
    assert prediction.shape == clean.shape
    trainer.compute_loss(prediction, sample)[0].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_cli_defaults():
    """References default to zero and no obsolete separate reference directory is needed."""
    args = entry.build_parser().parse_args([])
    assert args.use_ref == 0 and args.nerf_bands == 6


@pytest.mark.parametrize('count', [0, 2])
def test_train_checkpoint_ema_and_sample(paired_dirs, tmp_path, monkeypatch, count):
    """Run one epoch, resume training, load EMA, and reconstruct every patch file.

    Args:
        paired_dirs: Temporary paired NPY directories.
        tmp_path: Temporary directory for logs, checkpoints and reconstructions.
        monkeypatch: Pytest fixture supplying a small architecture preset.
        count: Reference count, exercising enabled and disabled workflows.
    """
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, 'Nano',
                        dict(hidden_size=16, num_groups=2, depth=2))
    args = make_args(paired_dirs, count)
    args.output_dir, args.log_id = str(tmp_path / 'train'), 'first'
    trainer = entry.ACDiTSeisDimReconNeRFTrainer(args)
    trainer.run()
    checkpoint = Path(trainer.checkpoint_dir) / 'checkpoint_epoch_00001'
    loaded = ACDiT2DWrapper.from_pretrained(checkpoint, device='cpu', use_ema=True)
    assert loaded.model.use_cross_attention == bool(count)
    args.ckpt, args.num_epochs, args.log_id = str(checkpoint), 2, 'resume'
    resumed = entry.ACDiTSeisDimReconNeRFTrainer(args)
    resumed.run()
    assert resumed.start_epoch == 1
    checkpoint = Path(resumed.checkpoint_dir) / 'checkpoint_epoch_00002'
    assert checkpoint.is_dir()

    sample_args = make_args(paired_dirs, count)
    sample_args.mode, sample_args.ckpt = 'sample', str(checkpoint)
    sample_args.output_dir, sample_args.log_id = str(tmp_path / 'sample'), 'recon'
    if not count:
        sample_args.input_dir = str(tmp_path / 'no_signal_needed')
    calls = []
    original_draw = entry.sample_reference_conditioning

    def record_draw(dataset, batch_size, args, device):
        """Record reference batch sizes while preserving the real draw and encoding.

        Args:
            dataset: Paired signal/coordinate reference pool.
            batch_size: Current number of targets.
            args: Reference count and NeRF configuration.
            device: Destination device.

        Returns:
            Reference conditioning produced by the real sampling helper.
        """
        calls.append(batch_size)
        return original_draw(dataset, batch_size, args, device)

    monkeypatch.setattr(entry, 'sample_reference_conditioning', record_draw)
    sampler = entry.ACDiTSeisDimReconNeRFSampler(sample_args)
    sampler.run()
    assert calls == ([2, 1, 2] if count else [])
    for name, patches in [('a.npy', 3), ('b.npy', 2)]:
        result = np.load(Path(sampler.output_dir) / name)
        assert result.shape == (patches, 1, 4, 6)
        assert np.isfinite(result).all()
