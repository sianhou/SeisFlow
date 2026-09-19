import importlib
import sys
from types import ModuleType, SimpleNamespace

import torch


def import_augmented_dit_module(monkeypatch):
    torchdiffeq = ModuleType("torchdiffeq")
    torchdiffeq.odeint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchdiffeq", torchdiffeq)
    return importlib.import_module("AugmentedDiTSeisDimReconNeRF")


def test_trainer_setup_model_forwards_augmented_dit_options(monkeypatch):
    module = import_augmented_dit_module(monkeypatch)
    trainer = module.AugmentedDiTSeisDimReconNeRFTrainer(
        SimpleNamespace(
            model_arch="T",
            patch_size=4,
            max_period=25,
            nerf_bands=1,
            nerf_include_input=True,
            upcast_attention=True,
            repa_lambda=0.0,
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

    monkeypatch.setattr(
        module,
        "build_augmented_dit_2d_wrapper",
        fake_builder,
    )

    trainer.setup_model()

    assert captured == {
        "model_arch": "T",
        "in_channels": 7,
        "out_channels": 1,
        "patch_size": 4,
        "num_classes": 1,
        "max_period": 25,
        "upcast_attention": True,
        "device": torch.device("cpu"),
    }


def test_parser_exposes_patch_size_and_max_period(monkeypatch):
    module = import_augmented_dit_module(monkeypatch)

    args = module.build_parser().parse_args([])

    assert args.model_arch == "T"
    assert args.patch_size == 4
    assert args.max_period == 10


def test_run_dispatches_sample_mode_to_augmented_sampler(monkeypatch):
    module = import_augmented_dit_module(monkeypatch)
    calls = []

    class FakeSampler:
        def __init__(self, args):
            calls.append(("init", args.mode))

        def run(self):
            calls.append(("run", "sample"))
            return "sampled"

    monkeypatch.setattr(
        module,
        "AugmentedDiTSeisDimReconNeRFSampler",
        FakeSampler,
    )
    args = module.build_parser().parse_args(["sample"])

    result = module.run(args)

    assert result == "sampled"
    assert calls == [("init", "sample"), ("run", "sample")]
