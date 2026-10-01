"""Exercise V4 coordinate routing, training, and checkpoint integration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import AugmentedDiTV4SeisDimReconNeRF as entry
from models.augmented_dit_2d_v4 import (
    AUGMENTED_DIT_2D_CONFIGS,
    AugmentedDiT2DModelV4,
    AugmentedDiT2DWrapperV4,
    build_augmented_dit_2d_wrapper_v4,
)
from models.wrapper import VelocityModel


@pytest.mark.parametrize("preset", AUGMENTED_DIT_2D_CONFIGS)
def test_builder_presets(preset):
    """Check architecture selection without allocating the full model weights."""
    with torch.device("meta"):
        wrapper = build_augmented_dit_2d_wrapper_v4(model_arch=preset)
    assert type(wrapper.model) is AugmentedDiT2DModelV4
    assert wrapper.model.in_channels == 1
    assert wrapper.model.coord_channels == 65
    for key, value in AUGMENTED_DIT_2D_CONFIGS[preset].items():
        assert getattr(wrapper.model, key) == value


@pytest.mark.parametrize("repa", [False, True])
def test_wrapper_routes_y_and_mask(repa):
    """Verify coordinate gradients and optional projected features against the model."""
    model = build_augmented_dit_2d_wrapper_v4(
        in_channels=1, coord_channels=3, out_channels=1, hidden_size=8,
        num_groups=2, depth=2, patch_size=2, max_period=37, device="cpu",
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "adaLN_modulation" in name or "final_layer.linear" in name:
                parameter.normal_(std=0.08)
    if repa:
        model.configure_repa(align_layer=1, projection_dim=4)
    x = torch.randn(2, 1, 4, 6)
    y = torch.randn(2, 3, 4, 6, requires_grad=True)
    t = torch.tensor([0.2, 0.8])
    mask = torch.eye(6, dtype=torch.bool)
    result = model(x, t, {"y": y, "mask": mask})
    expected = model.model(x, t, y, mask=mask, return_patch_feature_at=0 if repa else None)
    if repa:
        prediction, features = result
        torch.testing.assert_close(features, model.repa_projection(expected[1]))
        expected = expected[0]
    else:
        prediction = result
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    prediction.square().mean().backward()
    assert y.grad.abs().max() > 0


@pytest.mark.parametrize("channels", [1, 5])
@pytest.mark.parametrize("bands,include_input", [(0, True), (2, False), (6, True)])
@pytest.mark.parametrize("use_ema", [False, True])
def test_train_sample_resume(tmp_path, monkeypatch, channels, bands, include_input, use_ema):
    """Train small NPY batches, compare sampling conditions, and reload raw/EMA weights."""
    signal_dir, coord_dir = tmp_path / "train", tmp_path / "train_dim"
    signal_dir.mkdir()
    coord_dir.mkdir()
    rng = np.random.default_rng(4)
    signals = rng.uniform(-1, 1, (3, 4, 6)).astype(np.float32)
    coord_shape = (3, 4, 6) if channels == 1 else (3, channels, 4, 6)
    coordinates = rng.uniform(-1, 1, coord_shape).astype(np.float32)
    np.save(signal_dir / "shot.npy", signals)
    np.save(coord_dir / "shot.npy", coordinates)
    cli = ["--device", "cpu", "--model_arch", "Nano", "--patch_size", "2",
           "--input_dir", str(signal_dir), "--input_dim_dir", str(coord_dir),
           "--nerf_bands", str(bands), "--num_workers", "0", "--batch_size", "2"]
    if not include_input:
        cli.append("--no-nerf_include_input")
    if not use_ema:
        cli.append("--no-use_ema")
    args = entry.build_parser().parse_args(cli)
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, "Nano", {
        "hidden_size": 8, "num_groups": 2, "depth": 2,
    })
    trainer = entry.AugmentedDiTSeisDimReconNeRFTrainer(args)
    trainer.device = torch.device("cpu")
    trainer._setup_dataset()
    trainer._setup_model()
    trainer.setup_optimizer()
    batch = [torch.from_numpy(signals[:, None]), torch.from_numpy(
        coordinates[:, None] if channels == 1 else coordinates,
    )]
    clean, training_extra = trainer.preprocess_batch(batch)
    sampler = entry.AugmentedDiTSeisDimReconNeRFSampler(args)
    sampler.device = trainer.device
    noise, sampling_extra = sampler.preprocess_batch(coordinates)
    assert clean.shape == noise.shape == (3, 1, 4, 6)
    assert set(training_extra) == set(sampling_extra) == {"y"}
    torch.testing.assert_close(training_extra["y"], sampling_extra["y"], rtol=0, atol=0)
    assert trainer.model.model.coord_channels == channels * (2 * bands + int(include_input))
    for epoch in range(2):
        assert np.isfinite(trainer.train_one_epoch(epoch))
    assert trainer.model.model.y_embedder.proj.weight.grad.abs().max() > 0
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.save_pretrained(2)
    args.ckpt = str(trainer.checkpoint_dir / "checkpoint_epoch_00002")
    restored, epoch, state = AugmentedDiT2DWrapperV4.from_pretrained(
        args.ckpt, device="cpu", return_training_state=True,
    )
    assert epoch == 2 and state["optimizer"]["state"]
    t = torch.tensor([0.2, 0.5, 0.8])
    torch.testing.assert_close(
        restored(noise, t, sampling_extra), trainer.model(noise, t, sampling_extra),
        rtol=0, atol=0,
    )
    loaded = sampler.setup_model()
    if use_ema:
        trainer.ema.copy_to(trainer.model.model.parameters())
    torch.testing.assert_close(
        loaded(noise, t, sampling_extra), trainer.model(noise, t, sampling_extra),
        rtol=0, atol=0,
    )
    velocity = VelocityModel(loaded)(
        noise, torch.tensor(0.5), cfg_scale=0.0, label=None,
        concat_conditioning=sampling_extra,
    )
    assert velocity.shape == noise.shape and torch.isfinite(velocity).all()
    trainer.from_pretrained()
    assert trainer.start_epoch == 2


def test_repa_setup_without_external_teacher(monkeypatch):
    """Exercise inherited REPA loss using a local teacher stand-in without downloading weights."""
    class Teacher(torch.nn.Module):
        """Supply deterministic patch features in the DINO adapter's expected shape."""

        def __init__(self, model_name, hub_dir):
            """Accept the DINO model/cache arguments and expose a four-channel encoder."""
            super().__init__()
            self.encoder = SimpleNamespace(embed_dim=4)

        def forward(self, x):
            """Return six four-channel reference tokens for each input image x."""
            return torch.ones(x.shape[0], 6, 4, device=x.device)

    monkeypatch.setattr(entry, "DINOv2", Teacher)
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, "Nano", {
        "hidden_size": 8, "num_groups": 2, "depth": 2,
    })
    args = entry.build_parser().parse_args([
        "--model_arch", "Nano", "--patch_size", "2", "--nerf_bands", "0",
        "--repa_lambda", "0.1", "--repa_align_layer", "1",
    ])
    trainer = entry.AugmentedDiTSeisDimReconNeRFTrainer(args)
    trainer.device = torch.device("cpu")
    trainer.dataset = SimpleNamespace(dataset1=[torch.zeros(2, 4, 6)])
    model = trainer.setup_model()
    clean, extra = trainer.preprocess_batch((torch.randn(2, 1, 4, 6), torch.randn(2, 2, 4, 6)))
    sample = trainer.sample_path(clean)
    output = model(sample["x_t"], sample["t"], extra)
    loss, _, auxiliary = trainer.compute_loss(output, sample)
    assert torch.isfinite(loss) and auxiliary > 0
    loss.backward()
    assert trainer.repa_projection[-1].weight.grad.abs().max() > 0
