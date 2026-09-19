import pytest
import torch
from torch import nn
from diffusers.training_utils import EMAModel

from models.wrapper import (
    AugmentedDiT2DWrapper,
    DiTTransformer2DWrapper,
    PixelDiT2DWrapper,
    DiTTransformer2DModel,
)


@pytest.mark.parametrize("wrapper_cls", [
    DiTTransformer2DWrapper, PixelDiT2DWrapper, AugmentedDiT2DWrapper,
])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_resume_places_ema_on_model_device(tmp_path, monkeypatch, wrapper_cls, device):
    # Meta exercises non-CPU placement even on hosts without CUDA.
    (tmp_path / "ema").mkdir()
    source = nn.Linear(2, 2)
    saved_ema = EMAModel(source.parameters(), decay=0.95)
    saved_ema.optimization_step = 123
    ema = EMAModel(nn.Linear(2, 2).to(device).parameters())
    model_cls = (DiTTransformer2DModel if wrapper_cls is DiTTransformer2DWrapper
                 else wrapper_cls.model_cls)
    monkeypatch.setattr(model_cls, "from_pretrained", lambda *a, **k: source)
    monkeypatch.setattr(EMAModel, "from_pretrained", lambda *a, **k: saved_ema)

    wrapper = wrapper_cls.from_pretrained(tmp_path, device=device, ema=ema)

    assert ema.optimization_step == 123
    assert ema.decay == 0.95
    assert all(p.device == next(wrapper.model.parameters()).device
               for p in ema.shadow_params)
    if device == "cpu":
        for actual, expected in zip(ema.shadow_params, saved_ema.shadow_params):
            torch.testing.assert_close(actual, expected)
    ema.step(wrapper.model.parameters())
    assert ema.optimization_step == 124
