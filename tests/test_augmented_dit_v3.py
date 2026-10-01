"""Verify V3 adapters and consistent reference conditioning in training and sampling."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from AugmentedDiTV3SeisDimReconNeRF import (
    AugmentedDiTSeisDimReconNeRFTrainer,
    AugmentedDiTSeisDimReconNeRFSampler,
    build_parser,
)
from models.augmented_dit_2d_v3 import (
    AUGMENTED_DIT_2D_CONFIGS,
    AugmentedDiT2DModelV3,
    AugmentedDiT2DWrapperV3,
    build_augmented_dit_2d_wrapper_v3,
)
from models.nerf import encode_nerf_conditioning
from models.wrapper import BaseModelWrapper


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_wrapper_conditioning_and_gradients(cross, concat, labels, masked):
    """Compare wrapper/direct predictions and verify both reference channel groups train."""
    torch.manual_seed(17)
    wrapper = build_augmented_dit_2d_wrapper_v3(
        in_channels=3, out_channels=1, hidden_size=8, num_groups=2, depth=2,
        patch_size=2, num_classes=2, max_period=37, use_cross_attention=cross,
        upcast_attention=True, device="cpu",
    )
    with torch.no_grad():
        for name, parameter in wrapper.named_parameters():
            if "adaLN_modulation" in name or "final_layer.linear" in name:
                parameter.normal_(std=0.08)
    x = torch.randn(2, 1 if concat else 3, 4, 6)
    times = torch.tensor([0.2, 0.8])
    extra = {}
    if concat:
        extra["concat_conditioning"] = torch.randn(2, 2, 4, 6)
    if labels:
        extra["label"] = torch.tensor([0, 1])
    if masked:
        extra["mask"] = torch.eye(6, dtype=torch.bool)
    if cross:
        extra["r"] = torch.randn(2, 3, 4, 6, requires_grad=True)
    output = wrapper(x, times, extra or None)
    combined = torch.cat((x, extra["concat_conditioning"]), dim=1) if concat else x
    expected = wrapper.model(
        combined, times, extra.get("label", torch.zeros(2, dtype=torch.long)),
        r=extra.get("r"), mask=extra.get("mask"),
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output.shape == (2, 1, 4, 6)
    assert output.abs().max() > 0
    output.square().mean().backward()
    if cross:
        assert extra["r"].grad[:, :1].abs().max() > 0
        assert extra["r"].grad[:, 1:].abs().max() > 0
    assert isinstance(wrapper, BaseModelWrapper)
    assert wrapper.model.config.max_period == 37
    assert not hasattr(wrapper.model, "ref_patch_embedder")


@pytest.mark.parametrize("preset", AUGMENTED_DIT_2D_CONFIGS)
def test_builder_presets(preset):
    """Construct each architecture without allocating large parameter tensors."""
    with torch.device("meta"):
        wrapper = build_augmented_dit_2d_wrapper_v3(model_arch=preset)
    assert type(wrapper.model) is AugmentedDiT2DModelV3
    for key, value in AUGMENTED_DIT_2D_CONFIGS[preset].items():
        assert getattr(wrapper.model, key) == value


def make_patch_directories(tmp_path, channels, explicit_ref_dim):
    """Write two aligned NPY files per branch and return their directory paths.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        channels: Number of raw coordinate channels, one or two.
        explicit_ref_dim: Whether reference coordinates use a non-default directory.

    Returns:
        Mapping of CLI input names to directories containing three total patches.
    """
    directories = {
        "input_dir": tmp_path / "train",
        "input_dim_dir": tmp_path / "train_dim",
        "ref": tmp_path / "train_ref",
        "ref_dim": tmp_path / ("custom_coordinates" if explicit_ref_dim else "train_ref_dim"),
    }
    for directory in directories.values():
        directory.mkdir()
    rng = np.random.default_rng(7)
    for name, count in [("a.npy", 2), ("b.npy", 1)]:
        for key, directory in directories.items():
            shape = (count, 4, 6) if key in ("input_dir", "ref") or channels == 1 else (count, channels, 4, 6)
            np.save(directory / name, rng.uniform(-1, 1, shape).astype(np.float32))
    return directories


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("channels,include_input", [(1, False), (2, True)])
@pytest.mark.parametrize("explicit_ref_dim", [False, True])
def test_training_sampling_and_checkpoint_roundtrip(
        tmp_path, monkeypatch, cross, channels, include_input, explicit_ref_dim,
):
    """Exercise disk pairing, independent coordinates, optimization, EMA, and resume."""
    directories = make_patch_directories(tmp_path, channels, explicit_ref_dim)
    cli = ["--device", "cpu", "--model_arch", "Nano", "--patch_size", "2",
           "--nerf_bands", "2", "--batch_size", "2", "--num_workers", "0"]
    for key in ("input_dir", "input_dim_dir"):
        cli.extend([f"--{key}", str(directories[key])])
    if cross:
        cli.extend(["--ref", str(directories["ref"])])
        if explicit_ref_dim:
            cli.extend(["--ref_dim", str(directories["ref_dim"])])
    if not include_input:
        cli.append("--no-nerf_include_input")
    args = build_parser().parse_args(cli)
    monkeypatch.setitem(AUGMENTED_DIT_2D_CONFIGS, "Nano", {
        "hidden_size": 8, "num_groups": 2, "depth": 2,
    })
    trainer = AugmentedDiTSeisDimReconNeRFTrainer(args)
    trainer.device = torch.device("cpu")
    trainer._setup_dataset()
    trainer._setup_model()
    trainer.setup_optimizer()
    sampler = AugmentedDiTSeisDimReconNeRFSampler(args)
    sampler.device = trainer.device
    assert len(trainer.dataset) == 3
    batches = iter(DataLoader(trainer.dataset, batch_size=2, shuffle=False))
    for filename in ("a.npy", "b.npy"):
        clean, training_extra = trainer.preprocess_batch(next(batches))
        input_file = directories["input_dim_dir"] / filename
        inputs = np.load(input_file, mmap_mode="r")
        sampled_batch = sampler.load_input_batch(inputs, input_file, 0, len(inputs))
        noise, sampling_extra = sampler.preprocess_batch(sampled_batch)
        assert noise.shape == clean.shape
        assert sampling_extra.keys() == training_extra.keys()
        for key in training_extra:
            torch.testing.assert_close(training_extra[key], sampling_extra[key], atol=0, rtol=0)
        if cross:
            ref_data = torch.from_numpy(np.load(directories["ref"] / filename)).unsqueeze(1)
            ref_coords = torch.from_numpy(np.load(directories["ref_dim"] / filename))
            if channels == 1:
                ref_coords = ref_coords.unsqueeze(1)
            expected_reference = torch.cat((ref_data, encode_nerf_conditioning(ref_coords, args)), dim=1)
            torch.testing.assert_close(training_extra["r"], expected_reference)
            assert not torch.equal(training_extra["r"][:, 1:], training_extra["concat_conditioning"])
        prediction = trainer.model(noise, torch.full((len(inputs),), 0.5), extra=sampling_extra)
        assert prediction.shape == clean.shape

    epoch_loss = trainer.train_one_epoch(0)
    assert np.isfinite(epoch_loss)
    assert trainer.model.model.final_layer.linear.weight.abs().max() > 0
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.save_pretrained(1)
    args.ckpt = str(trainer.checkpoint_dir / "checkpoint_epoch_00001")
    restored, epoch, state = AugmentedDiT2DWrapperV3.from_pretrained(
        args.ckpt, return_training_state=True, device="cpu",
    )
    assert epoch == 1
    assert state["optimizer"]["state"]
    assert state["args"]["nerf_bands"] == 2
    times = torch.full((len(noise),), 0.5)
    torch.testing.assert_close(
        restored(noise, times, extra=sampling_extra),
        trainer.model(noise, times, extra=sampling_extra), atol=0, rtol=0,
    )
    sampled_model = sampler.setup_model()
    trainer.ema.copy_to(trainer.model.model.parameters())
    torch.testing.assert_close(
        sampled_model(noise, times, extra=sampling_extra),
        trainer.model(noise, times, extra=sampling_extra), atol=0, rtol=0,
    )
    trainer.from_pretrained()
    assert trainer.start_epoch == 1
