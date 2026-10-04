"""Check input-only concatenation, shared reference paths and V2 persistence."""

import pytest
import torch

from models.acdit2d_v2 import ACDiT2DModelV2, ACDiT2DWrapperV2
from models.pixeldit import AugmentedDiT2DModel


def make_model(active=True, **overrides):
    """Return a small V2 model for checking nontrivial outputs and gradients.

    Args:
        active: Randomize zero-initialized modulation/output weights when True.
        **overrides: Constructor overrides for ACDiT2DModelV2.

    Returns:
        V2 model with six concatenated input channels and one signal output.
    """
    torch.manual_seed(13)
    config = dict(in_channels=6, out_channels=1, hidden_size=16, num_groups=2,
                  depth=3, patch_size=2, num_classes=1)
    config.update(overrides)
    model = ACDiT2DModelV2(**config)
    if active:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'adaLN_modulation' in name or 'final_layer.linear' in name:
                    parameter.normal_(std=0.1)
    return model


@pytest.mark.parametrize('depth', [1, 4])
def test_no_reference_matches_augmented_dit_at_every_depth(depth):
    """Compare forward/backward with weight-matched AugmentedDiT at the given depth.

    Args:
        depth: Transformer layer count, including multiple layers to detect reinjection.
    """
    model = make_model(depth=depth)
    baseline = AugmentedDiT2DModel(
        in_channels=6, out_channels=1, hidden_size=16, num_groups=2,
        depth=depth, patch_size=2, num_classes=1,
    )
    for name in ('patch_embedder', 't_embedder', 'y_embedder', 'final_layer'):
        getattr(baseline, name).load_state_dict(getattr(model, name).state_dict())
    for split, original in zip(model.patch_blocks, baseline.patch_blocks):
        attention, mlp = split['attention'], split['mlp']
        original.norm1.load_state_dict(attention.norm.state_dict())
        original.attn.load_state_dict(attention.attn.state_dict())
        original.norm2.load_state_dict(mlp.norm.state_dict())
        original.mlp.load_state_dict(mlp.mlp.state_dict())
        with torch.no_grad():
            for name in ('weight', 'bias'):
                getattr(original.adaLN_modulation[0], name).copy_(torch.cat([
                    getattr(attention.adaLN_modulation[0], name),
                    getattr(mlp.adaLN_modulation[0], name),
                ]))
    inputs = torch.randn(2, 6, 4, 6, requires_grad=True)
    times, labels = torch.tensor([0.2, 0.8]), torch.zeros(2, dtype=torch.long)
    actual, expected = model(inputs, times, labels), baseline(inputs, times, labels)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=3e-6)


def test_target_and_references_have_no_coordinate_reinjection():
    """Zero gates leave all target/reference tokens unchanged through every layer."""
    model = make_model(active=False, use_cross_attention=True)
    inputs = [torch.randn(2, 6, 4, 6) for _ in range(3)]
    embedded = [model.embed_inputs(data)[0] for data in inputs]
    seen = [[] for _ in model.patch_blocks]
    handles = [block['attention'].register_forward_pre_hook(
        lambda module, args, index=index: seen[index].append(args[0].clone()),
    ) for index, block in enumerate(model.patch_blocks)]
    output = model(inputs[0], torch.tensor([0.3, 0.7]), torch.zeros(2, dtype=torch.long), r=inputs[1:])
    for handle in handles:
        handle.remove()
    assert not hasattr(model, 'coords_patch_embedder')
    assert torch.count_nonzero(output) == 0
    for layer_inputs in seen:
        assert len(layer_inputs) == 3
        for tokens, expected in zip(layer_inputs, embedded):
            torch.testing.assert_close(tokens, expected)


@pytest.mark.parametrize('upcast', [False, True])
@pytest.mark.parametrize('count', [1, 2])
def test_references_share_attention_before_cross_and_receive_gradients(upcast, count):
    """Check shared SA outputs feed cross-attention with independently sized references.

    Args:
        upcast: Whether attention computes in float32 under CPU autocast.
        count: Number of references with different spatial token grids.
    """
    model = make_model(use_cross_attention=True, upcast_attention=upcast)
    x = torch.randn(2, 6, 4, 6, requires_grad=True)
    references = [torch.randn(2, 6, 2 * (i + 1), 4, requires_grad=True) for i in range(count)]
    shared_outputs, cross_inputs = [], []
    sa_handle = model.patch_blocks[0]['attention'].register_forward_hook(
        lambda module, args, output: shared_outputs.append(output),
    )
    ca_handle = model.patch_blocks[0]['cross_attention'].register_forward_pre_hook(
        lambda module, args: cross_inputs.append(args[:2]),
    )
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(x, torch.tensor([0.2, 0.8]), torch.zeros(2, dtype=torch.long), r=references)
    sa_handle.remove()
    ca_handle.remove()
    torch.testing.assert_close(cross_inputs[0][0], shared_outputs[0])
    torch.testing.assert_close(cross_inputs[0][1], torch.cat(shared_outputs[1:], dim=1))
    assert output.shape == (2, 1, 4, 6) and torch.isfinite(output).all()
    output.float().square().mean().backward()
    for data in [x, *references]:
        assert torch.isfinite(data.grad).all()
        assert data.grad[:, :1].abs().sum() > 0
        assert data.grad[:, 1:].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('reference_form', ['tensor', 'list', 'tuple'])
def test_wrapper_concatenates_after_common_time_noising(reference_form):
    """Verify t=0/intermediate/1 mixes only signals, not coordinate channels.

    Args:
        reference_form: Container format for clean references, coordinates and noises.
    """
    model = make_model(use_cross_attention=True)
    wrapper = ACDiT2DWrapperV2(model)
    x = torch.randn(3, 1, 4, 6)
    time = torch.tensor([0.0, 0.35, 1.0])
    extra = {'x_coord': torch.randn(3, 5, 4, 6)}
    for key, channels in [('r', 1), ('r_noise', 1), ('r_coord', 5)]:
        values = [torch.randn(3, channels, 4, 6) for _ in range(2)]
        extra[key] = values[0] if reference_form == 'tensor' else (
            tuple(values) if reference_form == 'tuple' else values
        )
    captured = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append((args, kwargs)), with_kwargs=True,
    )
    actual = wrapper(x, time, extra)
    handle.remove()
    args, kwargs = captured[0]
    torch.testing.assert_close(args[0], torch.cat((x, extra['x_coord']), dim=1))
    references = [extra['r']] if reference_form == 'tensor' else extra['r']
    noises = [extra['r_noise']] if reference_form == 'tensor' else extra['r_noise']
    coords = [extra['r_coord']] if reference_form == 'tensor' else extra['r_coord']
    passed = [kwargs['r']] if reference_form == 'tensor' else kwargs['r']
    for reference, noise, coord, combined in zip(references, noises, coords, passed):
        expected_signal = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * reference
        torch.testing.assert_close(combined[:, :1], expected_signal)
        torch.testing.assert_close(combined[:, 1:], coord)
    torch.testing.assert_close(wrapper(x, time, extra), actual, atol=0, rtol=0)


def test_checkpoint_round_trip(tmp_path):
    """Save/load nonzero V2 predictions and the concatenated channel configuration.

    Args:
        tmp_path: Pytest temporary checkpoint directory.
    """
    wrapper = ACDiT2DWrapperV2(make_model(use_cross_attention=True))
    x, time = torch.randn(2, 1, 4, 6), torch.tensor([0.2, 0.8])
    extra = {'x_coord': torch.randn(2, 5, 4, 6), 'r': torch.randn_like(x),
             'r_noise': torch.randn_like(x), 'r_coord': torch.randn(2, 5, 4, 6)}
    expected = wrapper(x, time, extra)
    wrapper.save_pretrained(tmp_path)
    restored = ACDiT2DWrapperV2.from_pretrained(tmp_path, device='cpu')
    assert restored.model.config.in_channels == 6
    assert 'in_coords_channels' not in restored.model.config
    torch.testing.assert_close(restored(x, time, extra), expected, atol=0, rtol=0)
