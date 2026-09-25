from types import SimpleNamespace

import numpy as np
import pytest
import torch

import AugmentedDiTSeisDimReconNeRFDirect as direct
from models.pixeldit import AugmentedDiT2DModel
from models.wrapper import AugmentedDiT2DWrapper


def tiny_builder(**kwargs):
    return AugmentedDiT2DWrapper(AugmentedDiT2DModel(
        in_channels=kwargs['in_channels'], out_channels=1,
        num_groups=2, hidden_size=8, depth=2, patch_size=2, num_classes=1,
    ))


@pytest.mark.parametrize('bands', [0, 2])
def test_coordinate_only_training_and_checkpoint_inference(tmp_path, monkeypatch, bands):
    args = direct.build_parser().parse_args(['--device', 'cpu', '--nerf_bands', str(bands)])
    trainer = direct.AugmentedDiTSeisDimReconNeRFDirectTrainer(args)
    trainer.device = torch.device('cpu')
    trainer.dataset = SimpleNamespace(dataset1=[torch.zeros(5, 4, 6)])
    monkeypatch.setattr(direct, 'build_augmented_dit_2d_wrapper', tiny_builder)
    model = trainer.setup_model()
    target = torch.randn(2, 1, 4, 6)
    coordinates = torch.randn(2, 5, 4, 6)

    def forbid_noise(*a, **k):
        raise AssertionError('Direct regression must not draw FM noise/time')

    monkeypatch.setattr(torch, 'randn_like', forbid_noise)
    monkeypatch.setattr(torch, 'rand', forbid_noise)
    inputs, extra = trainer.preprocess_batch((target, coordinates))
    sample = trainer.sample_path(inputs)
    assert inputs.shape[1] == 5 * (1 + 2 * bands)
    assert model.model.config.in_channels == inputs.shape[1]
    assert torch.count_nonzero(sample['t']) == 0
    prediction = model(sample['x_t'], sample['t'], extra)
    total, loss, auxiliary = trainer.compute_loss(prediction, sample)
    torch.testing.assert_close(loss, torch.nn.functional.mse_loss(prediction, target))
    total.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    torch.optim.SGD(model.parameters(), lr=.01).step()

    other_inputs, _ = trainer.preprocess_batch((target + 100, coordinates))
    torch.testing.assert_close(inputs, other_inputs)
    ckpt = tmp_path / 'checkpoint'
    model.save_pretrained(ckpt)
    args.ckpt = str(ckpt)
    args.use_ema = False
    sampler = direct.AugmentedDiTSeisDimReconNeRFDirectSampler(args)
    sampler.device = torch.device('cpu')
    restored = sampler.setup_model().eval()
    sampler.setup_sampler()
    assert not hasattr(sampler, 'solver')
    test_inputs, test_extra = sampler.preprocess_batch(coordinates.numpy())
    torch.testing.assert_close(test_inputs, inputs)
    with torch.no_grad():
        expected = model.eval()(inputs, sample['t'], extra)
        actual = restored(test_inputs, sample['t'], test_extra)
    torch.testing.assert_close(actual, expected)


def test_file_inference_one_forward_per_batch_and_clipping(tmp_path):
    args = direct.build_parser().parse_args([
        'sample', '--device', 'cpu', '--nerf_bands', '0', '--batch_size', '2',
        '--clip_recon', '-1', '1',
    ])
    sampler = direct.AugmentedDiTSeisDimReconNeRFDirectSampler(args)
    sampler.device = torch.device('cpu')
    data = tmp_path / 'coordinates'
    data.mkdir()
    coordinates = np.full((3, 5, 4, 6), 2, dtype=np.float32)
    path = data / 'patches_0001.npy'
    np.save(path, coordinates)
    sampler.dataset = SimpleNamespace(data_path=data)
    sampler.rank_files = [path]
    sampler.output_dir = tmp_path / 'output'
    calls = []

    class Model(torch.nn.Module):
        def forward(self, x, t, extra=None):
            calls.append(x.shape[0])
            assert torch.count_nonzero(t) == 0
            assert x.shape[1] == 5
            return x[:, :1]

    sampler.model = Model()
    sampler.setup_sampler()
    sampler.sample_one_epoch()
    assert calls == [2, 1]
    result = np.load(sampler.output_dir / path.name)
    np.testing.assert_array_equal(result, np.ones((3, 1, 4, 6), dtype=np.float32))
