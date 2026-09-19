import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from diffusers import DiTTransformer2DModel
from diffusers.training_utils import EMAModel as EMA

from models.wrapper import DiTTransformer2DWrapper, TRAINING_STATE_NAME


def import_dit_module(monkeypatch):
    torchdiffeq = ModuleType("torchdiffeq")
    torchdiffeq.odeint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchdiffeq", torchdiffeq)
    return importlib.import_module("DiTSeisDimReconNeRF")


def test_trainer_setup_model_uses_dit_size_and_nerf_channels(monkeypatch):
    module = import_dit_module(monkeypatch)
    trainer = module.DiTSeisDimReconNeRFTrainer(
        SimpleNamespace(
            model_arch="DiT_T_4",
            input_size=64,
            nerf_bands=1,
            nerf_include_input=True,
            upcast_attention=True,
        )
    )
    trainer.dataset = SimpleNamespace(
        dataset1=[torch.zeros(2, 64, 64)]
    )
    trainer.device = torch.device("cpu")
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "build_dit_transformer_2d_wrapper", fake_builder)

    trainer.setup_model()

    assert captured["model_arch"] == "DiT_T_4"
    assert captured["sample_size"] == 64
    assert captured["in_channels"] == 7
    assert captured["out_channels"] == 1
    assert captured["num_embeds_ada_norm"] == 1
    assert captured["upcast_attention"] is True
    assert captured["device"] == torch.device("cpu")


def test_trainer_preprocess_batch_encodes_nerf_conditioning(monkeypatch):
    module = import_dit_module(monkeypatch)
    trainer = module.DiTSeisDimReconNeRFTrainer(
        SimpleNamespace(
            nerf_bands=1,
            nerf_include_input=True,
        )
    )
    trainer.device = torch.device("cpu")
    clean = torch.zeros(2, 1, 3, 4)
    conditioning = torch.zeros(2, 2, 3, 4)

    processed, extra = trainer.preprocess_batch((clean, conditioning))

    assert processed is clean
    assert extra["concat_conditioning"].shape == (2, 6, 3, 4)


def test_sampler_preprocess_batch_builds_noise_and_conditioning(monkeypatch):
    module = import_dit_module(monkeypatch)
    sampler = module.DiTSeisDimReconNeRFSampler(
        SimpleNamespace(
            nerf_bands=1,
            nerf_include_input=True,
        )
    )
    sampler.device = torch.device("cpu")
    batch = np.zeros((2, 2, 3, 4), dtype=np.float32)

    noise, extra = sampler.preprocess_batch(batch)

    assert noise.shape == (2, 1, 3, 4)
    assert noise.dtype == torch.float32
    assert extra["concat_conditioning"].shape == (2, 6, 3, 4)


def test_run_dispatches_sample_mode_to_sampler(monkeypatch):
    module = import_dit_module(monkeypatch)
    calls = []

    class FakeSampler:
        def __init__(self, args):
            calls.append(("init", args.mode))

        def run(self):
            calls.append(("run", "sample"))
            return "sampled"

    monkeypatch.setattr(module, "DiTSeisDimReconNeRFSampler", FakeSampler)
    args = module.build_parser().parse_args(["sample"])

    result = module.run(args)

    assert result == "sampled"
    assert calls == [("init", "sample"), ("run", "sample")]


def test_dit_sampling_loads_ema_without_training_state(tmp_path):
    model = DiTTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=2,
        out_channels=1,
        num_layers=1,
        norm_num_groups=2,
        sample_size=4,
        patch_size=2,
        num_embeds_ada_norm=1,
        norm_type="ada_norm_zero",
    )
    wrapper = DiTTransformer2DWrapper(model)
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
    loaded = DiTTransformer2DWrapper.from_pretrained(
        tmp_path,
        use_ema=True,
    )
    first_parameter = next(loaded.model.parameters())
    assert torch.allclose(
        first_parameter,
        torch.full_like(first_parameter, 0.25),
    )
