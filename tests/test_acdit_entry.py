"""Verify aligned references, flow-time noise and ACDiT train/sample integration."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import ACDiTSeisDimReconNeRF as entry
from models.acdit2d import ACDiT2DWrapper
from models.nerf import encode_nerf_conditioning
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS


@pytest.fixture
def paired_dirs(tmp_path):
    """Write five target patches with identifiable coordinates across two files.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        Signal and coordinate paths containing [P, 1, 4, 6] and [P, 5, 4, 6] arrays.
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


@pytest.fixture
def reference_dirs(paired_dirs, tmp_path):
    """Write two aligned reference slots with distinct signal/coordinate offsets.

    Args:
        paired_dirs: Target signal and coordinate paths.
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        Two (signal, coordinate) directory pairs matching target filenames/counts.
    """
    directories = []
    for slot in (1, 2):
        signal_dir, coord_dir = tmp_path / f'ref{slot}', tmp_path / f'ref{slot}_dim'
        signal_dir.mkdir()
        coord_dir.mkdir()
        for source in paired_dirs[0].glob('*.npy'):
            signal = np.load(source) + slot
            # Exercise the builder's channel-less signal file layout [P, H, W].
            np.save(signal_dir / source.name, signal[:, 0])
            np.save(coord_dir / source.name, np.load(paired_dirs[1] / source.name) + slot)
        directories.append((signal_dir, coord_dir))
    return directories


def make_args(paired_dirs, reference_dirs=()):
    """Return a CPU CLI namespace suitable for small integration tests.

    Args:
        paired_dirs: Target signal and coordinate paths.
        reference_dirs: Zero, one or two aligned signal/coordinate directory pairs.

    Returns:
        Parsed training arguments with a small batch and one training epoch.
    """
    argv = [
        '--input_dir', str(paired_dirs[0]), '--input_dim_dir', str(paired_dirs[1]),
        '--device', 'cpu', '--model_arch', 'Nano', '--patch_size', '2',
        '--batch_size', '2', '--num_workers', '0', '--num_epochs', '1',
        '--save_every_epochs', '1', '--nerf_bands', '1', '--solver_step_size', '0.5',
    ]
    for slot, (signal_dir, coord_dir) in enumerate(reference_dirs, start=1):
        argv.extend([f'--ref_dir{slot}', str(signal_dir), f'--ref_dim_dir{slot}', str(coord_dir)])
    return entry.build_parser().parse_args(argv)


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('include_input', [False, True])
def test_shuffled_references_preserve_target_alignment(paired_dirs, reference_dirs, count, include_input):
    """Shuffling preserves target/reference indices; encoding does not change pairing.

    Args:
        paired_dirs: Target NPY directories.
        reference_dirs: Two aligned reference directory pairs.
        count: Number of enabled reference slots.
        include_input: Whether NeRF encoding includes raw coordinates.
    """
    args = make_args(paired_dirs, reference_dirs[:count])
    args.nerf_include_input = include_input
    dataset = entry.AlignedReferencePatchDataset(*paired_dirs, entry.reference_directories(args))
    generator = torch.Generator().manual_seed(19)
    for data, coords, pairs in DataLoader(dataset, batch_size=2, shuffle=True, generator=generator):
        extra = entry.prepare_reference_conditioning(pairs, args, 'cpu')
        assert len(extra['r']) == len(extra['r_coord']) == len(extra['r_noise']) == count
        for slot in range(count):
            torch.testing.assert_close(extra['r'][slot], data + slot + 1)
            torch.testing.assert_close(extra['r_coord'][slot],
                                       encode_nerf_conditioning(coords + slot + 1, args))
            assert extra['r_noise'][slot].shape == data.shape
        if count == 2:
            assert not torch.equal(extra['r_noise'][0], extra['r_noise'][1])


@pytest.mark.parametrize('count', [0, 1, 2])
def test_trainer_preprocess_and_model(paired_dirs, reference_dirs, monkeypatch, count):
    """Exercise target-only velocity supervision with zero, one or two aligned references.

    Args:
        paired_dirs: Target NPY directories.
        reference_dirs: Two aligned reference directory pairs.
        monkeypatch: Fixture supplying a small architecture preset.
        count: Number of enabled reference slots.
    """
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, 'Nano',
                        dict(hidden_size=16, num_groups=2, depth=2))
    trainer = entry.ACDiTSeisDimReconNeRFTrainer(make_args(paired_dirs, reference_dirs[:count]))
    trainer.device = torch.device('cpu')
    trainer.dataset = trainer.setup_dataset()
    batch = next(iter(DataLoader(trainer.dataset, batch_size=2)))
    clean, extra = trainer.preprocess_batch(batch)
    torch.testing.assert_close(clean, batch[0])
    torch.testing.assert_close(extra['x_coord'], encode_nerf_conditioning(batch[1], trainer.args))
    if count:
        assert len(extra['r']) == count
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


def test_cli_reference_slots(paired_dirs, reference_dirs):
    """Directories determine enabled slots, including a standalone second reference.

    Args:
        paired_dirs: Target NPY directories.
        reference_dirs: Two aligned reference directory pairs.
    """
    args = make_args(paired_dirs)
    assert entry.reference_directories(args) == []
    assert not hasattr(args, 'use_ref') and not hasattr(args, 'ref_dir')
    args.ref_dir2, args.ref_dim_dir2 = map(str, reference_dirs[1])
    assert entry.reference_directories(args) == [(args.ref_dir2, args.ref_dim_dir2)]
    with pytest.raises(SystemExit):
        entry.build_parser().parse_args(['--use_ref', '2'])


@pytest.mark.parametrize('count', [0, 2])
def test_train_checkpoint_ema_and_sample(paired_dirs, reference_dirs, tmp_path, monkeypatch, count):
    """Resume training and reconstruct separate target files with fixed per-run reference noise.

    Args:
        paired_dirs: Target training directories.
        reference_dirs: Two aligned training reference directory pairs.
        tmp_path: Temporary directory for checkpoints and sampling inputs/outputs.
        monkeypatch: Fixture supplying a small model and recording sampler conditions.
        count: Reference count, exercising enabled and disabled workflows.
    """
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, 'Nano',
                        dict(hidden_size=16, num_groups=2, depth=2))
    args = make_args(paired_dirs, reference_dirs[:count])
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

    target_dir = tmp_path / 'valid_dim'
    target_dir.mkdir()
    np.save(target_dir / 'validation.npy', np.full((3, 5, 4, 6), -2, dtype=np.float32))
    valid_references = []
    for slot in range(1, count + 1):
        signal_dir, coord_dir = tmp_path / f'valid_ref{slot}', tmp_path / f'valid_ref{slot}_dim'
        signal_dir.mkdir()
        coord_dir.mkdir()
        data = np.broadcast_to(np.arange(3, dtype=np.float32)[:, None, None] + slot,
                               (3, 4, 6)).copy()
        np.save(signal_dir / 'validation.npy', data)
        np.save(coord_dir / 'validation.npy', np.repeat(data[:, None], 5, axis=1))
        valid_references.append((signal_dir, coord_dir))

    sample_args = make_args(paired_dirs, valid_references)
    sample_args.mode, sample_args.ckpt = 'sample', str(checkpoint)
    sample_args.output_dir, sample_args.log_id = str(tmp_path / 'sample'), 'recon'
    sample_args.input_dim_dir = str(target_dir)
    sample_args.input_dir = str(tmp_path / 'no_target_signal_needed')
    conditions, evaluations = [], []
    original_prepare = entry.prepare_reference_conditioning
    original_forward = ACDiT2DWrapper.forward

    def record_prepare(reference_pairs, args, device):
        """Record one conditioning object per batch before the ODE run.

        Args:
            reference_pairs: Aligned reference tensor pairs.
            args: NeRF settings.
            device: Destination device.

        Returns:
            The real helper's clean references, encoded coordinates and fixed noises.
        """
        result = original_prepare(reference_pairs, args, device)
        conditions.append(result)
        return result

    def record_forward(wrapper, x, timesteps, extra=None):
        """Record each ODE call's fixed reference tensors before normal inference.

        Args:
            wrapper: ACDiT wrapper receiving the call.
            x: ODE target state [B, 1, H, W].
            timesteps: Current flow times [B].
            extra: Coordinates and optional reference conditions.

        Returns:
            Target velocity [B, 1, H, W] from the original wrapper.
        """
        if count:
            for key in ('r', 'r_coord', 'r_noise'):
                assert extra[key] is conditions[-1][key]
            evaluations.append((len(conditions) - 1, timesteps.clone(),
                                [noise.clone() for noise in extra['r_noise']]))
        return original_forward(wrapper, x, timesteps, extra)

    monkeypatch.setattr(entry, 'prepare_reference_conditioning', record_prepare)
    monkeypatch.setattr(ACDiT2DWrapper, 'forward', record_forward)
    sampler = entry.ACDiTSeisDimReconNeRFSampler(sample_args)
    sampler.run()
    assert len(conditions) == 2
    if count:
        for slot in range(count):
            signals = torch.cat([condition['r'][slot] for condition in conditions])
            torch.testing.assert_close(signals[:, 0, 0, 0], torch.arange(3).float() + slot + 1)
        assert {index for index, _, _ in evaluations} == {0, 1}
        assert any(time.max() > 0 for _, time, _ in evaluations)
        for index, _, noises in evaluations:
            for slot in range(count):
                torch.testing.assert_close(noises[slot], conditions[index]['r_noise'][slot])
    assert sampler.dataset.data_path == target_dir
    output_files = list(Path(sampler.output_dir).glob('*.npy'))
    assert [file.name for file in output_files] == ['validation.npy']
    result = np.load(output_files[0])
    assert result.shape == (3, 1, 4, 6)
    assert np.isfinite(result).all()
