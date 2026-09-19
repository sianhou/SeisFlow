import pytest
import torch
from diffusers.training_utils import EMAModel as EMA

import models.pixeldit as pixeldit
from models.pixeldit import PixDiT, RotaryAttention, precompute_freqs_cis_2d
from models.wrapper import PixelDiT2DWrapper, TRAINING_STATE_NAME


def build_tiny_pixeldit():
    return PixDiT(
        in_channels=2,
        out_channels=1,
        num_groups=2,
        hidden_size=8,
        pixel_hidden_size=4,
        patch_depth=2,
        pixel_depth=1,
        patch_size=2,
        num_classes=1,
    )


def test_pixeldit_wrapper_returns_tensor_when_repa_is_disabled():
    wrapper = PixelDiT2DWrapper(build_tiny_pixeldit())
    inputs = torch.randn(2, 2, 4, 4)
    timesteps = torch.rand(2)

    output = wrapper(inputs, timesteps)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 1, 4, 4)


def test_pixeldit_wrapper_projects_repa_feature_inside_forward():
    wrapper = PixelDiT2DWrapper(build_tiny_pixeldit())
    wrapper.configure_repa(align_layer=1, projection_dim=6)
    inputs = torch.randn(2, 2, 4, 4)
    timesteps = torch.rand(2)

    prediction, projected_feature = wrapper(inputs, timesteps)
    projected_feature.square().mean().backward()

    assert prediction.shape == (2, 1, 4, 4)
    assert projected_feature.shape == (2, 4, 6)
    assert "repa_projection.0.weight" in dict(wrapper.named_parameters())
    assert all(
        parameter.grad is not None
        for parameter in wrapper.repa_projection.parameters()
    )


@pytest.mark.parametrize("align_layer", [0, 3])
def test_pixeldit_wrapper_rejects_invalid_repa_layer(align_layer):
    wrapper = PixelDiT2DWrapper(build_tiny_pixeldit())

    with pytest.raises(ValueError, match="repa_align_layer"):
        wrapper.configure_repa(
            align_layer=align_layer,
            projection_dim=6,
        )


def test_upcast_attention_propagates_to_all_pixeldit_blocks():
    model = PixDiT(
        in_channels=2,
        out_channels=1,
        num_groups=2,
        hidden_size=8,
        pixel_hidden_size=4,
        patch_depth=2,
        pixel_depth=2,
        patch_size=2,
        num_classes=1,
        upcast_attention=True,
    )

    assert model.config.upcast_attention is True
    assert all(block.attn.upcast_attention for block in model.patch_blocks)
    assert all(block.attn.upcast_attention for block in model.pixel_blocks)


def test_upcast_attention_calls_sdpa_in_float32(monkeypatch):
    captured_dtypes = []

    def fake_sdpa(q, k, v, **kwargs):
        del q, k, kwargs
        captured_dtypes.append(v.dtype)
        return v

    monkeypatch.setattr(pixeldit, "scaled_dot_product_attention", fake_sdpa)
    attention = RotaryAttention(
        dim=8,
        num_heads=2,
        upcast_attention=True,
    )
    inputs = torch.randn(1, 4, 8)
    position = precompute_freqs_cis_2d(dim=4, height=2, width=2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = attention(inputs, position, mask=None)

    assert captured_dtypes == [torch.float32]
    assert output.shape == inputs.shape


def test_pixeldit_sampling_loads_ema_without_training_state(tmp_path):
    model = PixDiT(
        in_channels=2,
        out_channels=1,
        num_groups=2,
        hidden_size=8,
        pixel_hidden_size=4,
        patch_depth=1,
        pixel_depth=1,
        patch_size=2,
        num_classes=1,
    )
    wrapper = PixelDiT2DWrapper(model)
    ema = EMA(
        model.parameters(),
        decay=0.9,
        model_cls=type(model),
        model_config=model.config,
    )
    for shadow_parameter in ema.shadow_params:
        shadow_parameter.fill_(0.25)

    wrapper.save_pretrained(tmp_path, ema=ema)

    assert not (tmp_path / TRAINING_STATE_NAME).exists()
    loaded = PixelDiT2DWrapper.from_pretrained(
        tmp_path,
        use_ema=True,
    )
    first_parameter = next(loaded.model.parameters())
    assert loaded.repa_projection is None
    assert loaded.repa_align_index is None
    assert torch.allclose(
        first_parameter,
        torch.full_like(first_parameter, 0.25),
    )
