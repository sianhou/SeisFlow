import pytest
import torch

from models.pixeldit import AugmentedDiT2DModel, TimestepConditioner
from models.wrapper import (
    AugmentedDiT2DWrapper,
    build_augmented_dit_2d_wrapper,
)


def build_tiny_augmented_dit(**kwargs):
    config = {
        "in_channels": 3,
        "out_channels": 1,
        "num_groups": 2,
        "hidden_size": 8,
        "depth": 2,
        "patch_size": 2,
        "num_classes": 1,
    }
    config.update(kwargs)
    return AugmentedDiT2DModel(**config)


def test_augmented_dit_supports_rectangular_inputs():
    model = build_tiny_augmented_dit()
    inputs = torch.randn(2, 3, 4, 6)
    timesteps = torch.rand(2)
    labels = torch.zeros(2, dtype=torch.long)

    output = model(inputs, timesteps, labels)

    assert output.shape == (2, 1, 4, 6)


def test_augmented_dit_wrapper_concatenates_conditioning():
    wrapper = AugmentedDiT2DWrapper(build_tiny_augmented_dit())
    inputs = torch.randn(2, 1, 4, 4)
    conditioning = torch.randn(2, 2, 4, 4)

    output = wrapper(
        inputs,
        torch.rand(2),
        {"concat_conditioning": conditioning},
    )

    assert output.shape == (2, 1, 4, 4)


def test_augmented_dit_returns_repa_feature():
    wrapper = AugmentedDiT2DWrapper(build_tiny_augmented_dit())
    wrapper.configure_repa(align_layer=1, projection_dim=6)

    prediction, projected_feature = wrapper(
        torch.randn(2, 3, 4, 4),
        torch.rand(2),
    )

    assert prediction.shape == (2, 1, 4, 4)
    assert projected_feature.shape == (2, 4, 6)


def test_max_period_changes_timestep_embedding():
    short_period = TimestepConditioner(8, max_period=10)
    long_period = TimestepConditioner(8, max_period=10000)
    long_period.load_state_dict(short_period.state_dict())
    timesteps = torch.tensor([0.25, 0.75])

    short_embedding = short_period(timesteps)
    long_embedding = long_period(timesteps)

    assert not torch.allclose(short_embedding, long_embedding)


def test_max_period_is_preserved_by_save_and_load(tmp_path):
    wrapper = AugmentedDiT2DWrapper(
        build_tiny_augmented_dit(max_period=37)
    )
    wrapper.save_pretrained(tmp_path)

    restored = AugmentedDiT2DWrapper.from_pretrained(tmp_path)

    assert restored.model.config.max_period == 37
    assert restored.model.t_embedder.max_period == 37


def test_augmented_dit_builder_forwards_max_period():
    wrapper = build_augmented_dit_2d_wrapper(
        model_arch="T",
        in_channels=3,
        out_channels=1,
        num_groups=2,
        hidden_size=8,
        depth=1,
        patch_size=2,
        num_classes=1,
        max_period=25,
    )

    assert wrapper.model.config.max_period == 25


def test_augmented_dit_rejects_incompatible_rope_head_dimension():
    with pytest.raises(ValueError, match="divisible by 4"):
        build_tiny_augmented_dit(hidden_size=12, num_groups=2)
