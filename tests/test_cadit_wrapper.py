"""Exercise CADiT conditioning, architecture construction, and base checkpoints."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.training_utils import EMAModel

from core.training.amp_scaler import AMPGradScaler
from models.cadit2d import CADiT2DModel, CADiT2DWrapper, build_cadit_2d_wrapper
from models.wrapper import AUGMENTED_DIT_2D_CONFIGS, BaseModelWrapper


def build_tiny_wrapper(cross=False):
    """Return a small CADiT wrapper with nonzero gates and optional cross attention."""
    torch.manual_seed(17)
    wrapper = build_cadit_2d_wrapper(
        in_channels=3, out_channels=1, num_groups=2, hidden_size=8,
        depth=2, patch_size=2, num_classes=3, max_period=37,
        use_cross_attention=cross,
    )
    with torch.no_grad():
        for name, parameter in wrapper.named_parameters():
            if "adaLN_modulation" in name or "final_layer.linear" in name:
                parameter.normal_(std=0.08)
    return wrapper


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_conditioning_matches_direct_model(cross, concat, labels, masked):
    """Compare wrapper/direct output for each reference, concat, label and mask flag."""
    wrapper = build_tiny_wrapper(cross)
    inputs = torch.randn(2, 1 if concat else 3, 4, 6)
    timesteps = torch.rand(2)
    extra = {}
    if concat:
        extra["concat_conditioning"] = torch.randn(2, 2, 4, 6)
    if labels:
        extra["label"] = torch.tensor([1, 3])
    if masked:
        extra["mask"] = torch.eye(6, dtype=torch.bool)
    if cross:
        extra["r"] = torch.randn(2, 1, 4, 6, requires_grad=True)

    prediction = wrapper(inputs, timesteps, extra or None)
    combined = torch.cat((inputs, extra["concat_conditioning"]), dim=1) if concat else inputs
    expected = wrapper.model(
        combined, timesteps, extra.get("label", torch.zeros(2, dtype=torch.long)),
        r=extra.get("r"), mask=extra.get("mask"),
    )
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    assert prediction.shape == (2, 1, 4, 6)
    assert prediction.abs().max() > 0
    prediction.square().mean().backward()
    if cross:
        assert extra["r"].grad.abs().max() > 0
    assert isinstance(wrapper, BaseModelWrapper)
    assert not hasattr(wrapper, "repa_projection")


@pytest.mark.parametrize("name", AUGMENTED_DIT_2D_CONFIGS)
def test_builder_uses_presets(name):
    """Construct full preset name on meta and verify its architecture and defaults."""
    with torch.device("meta"):
        wrapper = build_cadit_2d_wrapper(model_arch=name, device="meta")
    assert type(wrapper) is CADiT2DWrapper
    assert type(wrapper.model) is CADiT2DModel
    for key, value in AUGMENTED_DIT_2D_CONFIGS[name].items():
        assert getattr(wrapper.model, key) == value
    assert wrapper.model.out_channels == 4
    assert wrapper.model.use_cross_attention is False
    assert next(wrapper.parameters()).device.type == "meta"


def test_builder_forwards_overrides():
    """Verify every explicit constructor override and the destination device."""
    config = dict(in_channels=5, out_channels=2, num_groups=3, hidden_size=24,
                  depth=1, patch_size=3, num_classes=7, max_period=25,
                  upcast_attention=True, use_cross_attention=True)
    wrapper = build_cadit_2d_wrapper(model_arch="XL", device="cpu", **config)
    for key, value in config.items():
        assert getattr(wrapper.model.config, key) == value
    assert next(wrapper.parameters()).device.type == "cpu"


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("use_ema", [False, True])
def test_checkpoint_restores_training_state(tmp_path, cross, use_ema):
    """Round-trip raw/EMA weights and supplied training states for each cross mode."""
    wrapper = build_tiny_wrapper(cross)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = AMPGradScaler(enabled=False, device="cpu")
    ema = EMAModel(wrapper.model.parameters(), model_cls=CADiT2DModel,
                   model_config=wrapper.model.config)
    inputs, times = torch.randn(2, 3, 4, 6), torch.rand(2)
    extra = {"r": torch.randn(2, 1, 4, 6)} if cross else None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        wrapper(inputs, times, extra).square().mean().backward()
        optimizer.step()
        scheduler.step()
        ema.step(wrapper.model.parameters())
    wrapper.save_pretrained(tmp_path, optimizer=optimizer, lr_scheduler=scheduler,
                            scaler=scaler, args=SimpleNamespace(seed=17), epoch=2, ema=ema)
    optimizer.load_state_dict({**optimizer.state_dict(), "state": {}})
    restored, epoch, state = CADiT2DWrapper.from_pretrained(
        tmp_path, optimizer=optimizer, lr_scheduler=scheduler, scaler=scaler,
        device="cpu", return_training_state=True, use_ema=use_ema,
        low_cpu_mem_usage=False,
    )
    assert isinstance(restored.model, CADiT2DModel)
    assert epoch == 2 and state["args"] == {"seed": 17}
    assert optimizer.state and scheduler.state_dict() == state["lr_scheduler"]
    assert restored.model.t_embedder.max_period == 37
    if use_ema:
        ema.copy_to(wrapper.model.parameters())
    torch.testing.assert_close(wrapper(inputs, times, extra), restored(inputs, times, extra), rtol=0, atol=0)


def test_model_only_checkpoint(tmp_path):
    """Load a model-only checkpoint without requiring a training state or EMA file."""
    wrapper = build_tiny_wrapper()
    wrapper.save_pretrained(tmp_path)
    restored = CADiT2DWrapper.from_pretrained(tmp_path, low_cpu_mem_usage=False)
    inputs, times = torch.randn(2, 3, 4, 6), torch.rand(2)
    torch.testing.assert_close(wrapper(inputs, times), restored(inputs, times), rtol=0, atol=0)
