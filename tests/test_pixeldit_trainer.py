import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def import_pixeldit_module(monkeypatch):
    torchdiffeq = ModuleType("torchdiffeq")
    torchdiffeq.odeint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchdiffeq", torchdiffeq)
    return importlib.import_module("PixelDiTSeisDimReconNeRF")


def import_pixeldit_trainer(monkeypatch):
    return import_pixeldit_module(monkeypatch).PixelDiTSeisDimReconNeRFTrainer


def test_repa_auxiliary_loss_is_weighted(monkeypatch):
    trainer_class = import_pixeldit_trainer(monkeypatch)
    trainer = trainer_class(SimpleNamespace(repa_lambda=0.5))
    trainer.repa_projected_feature = torch.tensor([[[1.0, 0.0]]])
    trainer.repa_clean_images = torch.zeros(1, 1, 1, 1)
    trainer.dino = lambda inputs: torch.tensor(
        [[[0.0, 1.0]]],
        device=inputs.device,
    )

    auxiliary_loss = trainer.compute_auxiliary_loss()

    # Orthogonal source/target features have cosine loss 1.0.
    assert float(auxiliary_loss) == pytest.approx(0.5)
    assert trainer.repa_projected_feature is None


def test_repa_compute_loss_uses_model_forward_projection(monkeypatch):
    trainer_class = import_pixeldit_trainer(monkeypatch)
    trainer = trainer_class(SimpleNamespace(repa_lambda=0.5))
    trainer.repa_clean_images = torch.zeros(1, 1, 1, 1)
    trainer.dino = lambda inputs: torch.tensor(
        [[[0.0, 1.0]]],
        device=inputs.device,
    )
    prediction = torch.zeros(1, 1, 1, 1, requires_grad=True)
    projected_feature = torch.tensor(
        [[[1.0, 0.0]]],
        requires_grad=True,
    )
    sample = {"velocity": torch.zeros_like(prediction)}

    total_loss, flow_loss, auxiliary_loss = trainer.compute_loss(
        (prediction, projected_feature),
        sample,
    )
    total_loss.backward()

    assert float(flow_loss.detach()) == pytest.approx(0.0)
    assert float(auxiliary_loss.detach()) == pytest.approx(0.5)
    assert projected_feature.grad is not None


def test_sampler_preprocess_batch_builds_noise_and_conditioning(monkeypatch):
    module = import_pixeldit_module(monkeypatch)
    sampler = module.PixelDiTSeisDimReconNeRFSampler(
        SimpleNamespace(
            nerf_bands=1,
            nerf_include_input=True,
        )
    )
    sampler.device = torch.device("cpu")
    batch = np.zeros((2, 3, 4), dtype=np.float32)

    noise, extra = sampler.preprocess_batch(batch)

    assert noise.shape == (2, 1, 3, 4)
    assert noise.dtype == torch.float32
    assert extra["concat_conditioning"].shape == (2, 3, 3, 4)
