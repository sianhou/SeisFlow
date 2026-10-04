"""Exercise V2 training, resume, EMA and file-based reference reconstruction."""

from pathlib import Path

import numpy as np
import pytest
import torch

import ACDiTSeisDimReconNeRF2 as entry
from models.acdit2d_v2 import ACDiT2DWrapperV2
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS


@pytest.mark.parametrize('count', [0, 1, 2])
@pytest.mark.parametrize('bands,include_input', [(0, True), (1, False), (1, True)])
def test_v2_train_resume_ema_and_sample(tmp_path, monkeypatch, count, bands, include_input):
    """Run the complete V2 workflow while preserving independent fixed reference noise.

    Args:
        tmp_path: Temporary data/checkpoint/output root.
        monkeypatch: Fixture setting a small architecture and recording ODE conditions.
        count: Number of enabled aligned reference slots.
        bands: NeRF frequency-band count.
        include_input: Whether encoded coordinates retain the raw channels.
    """
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, 'Nano',
                        dict(hidden_size=16, num_groups=2, depth=2))
    rng = np.random.default_rng(17)
    for name, shape in [('data', (3, 4, 6)), ('coords', (3, 5, 4, 6))]:
        directory = tmp_path / name
        directory.mkdir()
        np.save(directory / 'patches_0001.npy', rng.normal(size=shape).astype(np.float32))
    argv = ['--input_dir', str(tmp_path / 'data'), '--input_dim_dir', str(tmp_path / 'coords'),
            '--device', 'cpu', '--model_arch', 'Nano', '--patch_size', '2',
            '--batch_size', '2', '--num_workers', '0', '--num_epochs', '1',
            '--save_every_epochs', '1', '--nerf_bands', str(bands), '--solver_step_size', '0.5',
            '--output_dir', str(tmp_path / 'train'), '--log_id', 'first']
    if not include_input:
        argv.append('--no-nerf_include_input')
    for slot in range(1, count + 1):
        for suffix, shape in [('', (3, 4, 6)), ('_dim', (3, 5, 4, 6))]:
            directory = tmp_path / f'ref{slot}{suffix}'
            directory.mkdir()
            np.save(directory / 'patches_0001.npy', rng.normal(size=shape).astype(np.float32))
        argv.extend([f'--ref_dir{slot}', str(tmp_path / f'ref{slot}'),
                     f'--ref_dim_dir{slot}', str(tmp_path / f'ref{slot}_dim')])
    args = entry.build_parser().parse_args(argv)
    trainer = entry.ACDiTSeisDimReconNeRF2Trainer(args)
    trainer.run()
    checkpoint = Path(trainer.checkpoint_dir) / 'checkpoint_epoch_00001'
    loaded = ACDiT2DWrapperV2.from_pretrained(checkpoint, device='cpu', use_ema=True)
    expected_channels = 1 + 5 * (2 * bands + int(include_input))
    assert loaded.model.in_channels == expected_channels
    assert loaded.model.out_channels == 1
    assert loaded.model.use_cross_attention == bool(count)
    args.ckpt, args.num_epochs, args.log_id = str(checkpoint), 2, 'resume'
    resumed = entry.ACDiTSeisDimReconNeRF2Trainer(args)
    resumed.run()
    assert resumed.start_epoch == 1
    args.ckpt = str(Path(resumed.checkpoint_dir) / 'checkpoint_epoch_00002')
    args.mode, args.output_dir, args.log_id = 'sample', str(tmp_path / 'sample'), 'recon'
    args.input_dir = str(tmp_path / 'no_target_signal_required')
    evaluations = []
    original_forward = ACDiT2DWrapperV2.forward

    def record_forward(wrapper, x, timesteps, extra=None):
        """Record ODE times and reference conditions before normal V2 prediction.

        Args:
            wrapper: V2 wrapper called by the ODE solver.
            x: Target signal state [B, 1, H, W].
            timesteps: Solver flow times [B].
            extra: Encoded coordinates and optional clean references/fixed noises.

        Returns:
            Target velocity [B, 1, H, W] from the original forward method.
        """
        evaluations.append((timesteps.clone(), extra))
        return original_forward(wrapper, x, timesteps, extra)

    monkeypatch.setattr(ACDiT2DWrapperV2, 'forward', record_forward)
    sampler = entry.ACDiTSeisDimReconNeRF2Sampler(args)
    sampler.run()
    assert any(time.max() > 0 for time, _ in evaluations)
    for batch_size in (2, 1):
        conditions = [extra for time, extra in evaluations if time.numel() == batch_size]
        for extra in conditions:
            assert extra['x_coord'].shape[1] == expected_channels - 1
            if count:
                for key in ('r', 'r_coord', 'r_noise'):
                    assert extra[key] is conditions[0][key]
                if count == 2:
                    assert not torch.equal(extra['r_noise'][0], extra['r_noise'][1])
    result = np.load(Path(sampler.output_dir) / 'patches_0001.npy')
    assert result.shape == (3, 1, 4, 6)
    assert np.isfinite(result).all()


def test_v2_cli_names_and_reference_arguments():
    """V2 help identifies its own entry point and retains the two directory pairs."""
    parser = entry.build_parser()
    help_text = parser.format_help()
    assert 'ACDiTSeisDimReconNeRF2.py' in help_text
    assert 'ACDiTSeisDimReconNeRF.py' not in help_text
    args = parser.parse_args(['--ref_dir2', 'reference', '--ref_dim_dir2', 'coords'])
    assert args.ref_dir1 is None and args.ref_dir2 == 'reference'
    assert not hasattr(args, 'use_ref')
