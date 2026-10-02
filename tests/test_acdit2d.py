"""Exercise repeated coordinate injection, self-attention, and ACDiT checkpoints."""

import pytest
import torch

from models.acdit2d import ACDiT2DModel, ACDiTCrossAttnBlock, ACDiTMLPBlock, ACDiTSelfAttnBlock
from models.pixeldit import AugmentedDiT2DModel, AugmentedDiTBlock, precompute_freqs_cis_2d


def make_model(**overrides):
    """Return a small model with active residuals to expose conditioning behavior.

    Args:
        overrides: Constructor overrides passed to ACDiT2DModel.

    Returns:
        ACDiT2DModel with nonzero modulation and output weights.
    """
    torch.manual_seed(13)
    config = dict(in_channels=1, in_coords_channels=5, out_channels=2,
                  hidden_size=16, num_groups=2, depth=3, patch_size=2,
                  num_classes=2)
    config.update(overrides)
    model = ACDiT2DModel(**config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'adaLN_modulation' in name or 'final_layer.linear' in name:
                parameter.normal_(std=0.1)
    return model


def copy_split_block_to_original(split, original):
    """Copy two independent components into the original block for equivalence tests.

    Args:
        split: ModuleDict containing attention and MLP components of width D.
        original: AugmentedDiTBlock receiving the same weights, including the
            concatenated modulation projection [6D, D] and bias [6D].
    """
    attention, mlp = split['attention'], split['mlp']
    original.norm1.load_state_dict(attention.norm.state_dict())
    original.attn.load_state_dict(attention.attn.state_dict())
    original.norm2.load_state_dict(mlp.norm.state_dict())
    original.mlp.load_state_dict(mlp.mlp.state_dict())
    with torch.no_grad():
        original.adaLN_modulation[0].weight.copy_(torch.cat([
            attention.adaLN_modulation[0].weight, mlp.adaLN_modulation[0].weight,
        ]))
        original.adaLN_modulation[0].bias.copy_(torch.cat([
            attention.adaLN_modulation[0].bias, mlp.adaLN_modulation[0].bias,
        ]))


def make_inputs(patch=2, batch=2, channels=1, coords=5):
    """Return data and coordinates on a rectangular patch grid for a model call.

    Args:
        patch: Token patch side length.
        batch: Batch size.
        channels: Number of data channels.
        coords: Number of coordinate feature channels.

    Returns:
        Keyword arguments with data/coordinates [B, C, 2p, 3p], times and labels [B].
    """
    return dict(
        x=torch.randn(batch, channels, 2 * patch, 3 * patch, requires_grad=True),
        x_coord=torch.randn(batch, coords, 2 * patch, 3 * patch, requires_grad=True),
        t=torch.linspace(0.1, 0.9, batch), y=torch.zeros(batch, dtype=torch.long),
    )


@pytest.mark.parametrize('patch,channels,coords,depth', [
    (1, 1, 5, 1), (2, 2, 3, 2), (4, 1, 5, 3),
])
@pytest.mark.parametrize('upcast', [False, True])
def test_shapes_and_gradients(patch, channels, coords, depth, upcast):
    """Exercise rectangular grids and gradients through data and coordinates.

    Args:
        patch: Token patch side length.
        channels: Input data channel count.
        coords: Coordinate feature channel count.
        depth: Encoder depth.
        upcast: Enable float32 attention computation.
    """
    model = make_model(in_channels=channels, in_coords_channels=coords,
                       depth=depth, patch_size=patch,
                       upcast_attention=upcast)
    inputs = make_inputs(patch, channels=channels, coords=coords)
    output = model(**inputs)
    assert output.shape == (2, 2, 2 * patch, 3 * patch)
    assert torch.isfinite(output).all() and output.abs().max() > 0
    output.square().mean().backward()
    for tensor in [inputs['x'], inputs['x_coord']]:
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_single_block_matches_concat_model():
    """Verify one coordinate injection equals a weight-matched single-block concat model."""
    model = make_model(depth=1)
    baseline = AugmentedDiT2DModel(
        in_channels=6, out_channels=2, hidden_size=16, num_groups=2,
        depth=1, patch_size=2, num_classes=2,
    )
    for name in ['t_embedder', 'y_embedder', 'final_layer']:
        getattr(baseline, name).load_state_dict(getattr(model, name).state_dict())
    copy_split_block_to_original(model.patch_blocks[0], baseline.patch_blocks[0])
    with torch.no_grad():
        baseline.patch_embedder.proj.weight.copy_(torch.cat([
            model.x_patch_embedder.proj.weight,
            model.coords_patch_embedder.proj.weight,
        ], dim=1))
        baseline.patch_embedder.proj.bias.copy_(
            model.x_patch_embedder.proj.bias + model.coords_patch_embedder.proj.bias,
        )
    inputs = make_inputs()
    actual = model(**inputs)
    expected = baseline(torch.cat([inputs['x'], inputs['x_coord']], dim=1),
                        inputs['t'], inputs['y'])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), (inputs['x'], inputs['x_coord']))
    expected_grad = torch.autograd.grad(expected.square().sum(), (inputs['x'], inputs['x_coord']))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('depth', [1, 4])
def test_coordinate_injection_at_every_block(depth):
    """With identity residual blocks, coordinates accumulate once per block.

    Args:
        depth: Number of blocks over which coordinate injections accumulate.
    """
    model = ACDiT2DModel(in_channels=1, hidden_size=16, num_groups=2,
                         depth=depth, patch_size=2, num_classes=2)
    inputs = make_inputs()
    x_tokens, coord_tokens, _ = model.embed_inputs(inputs['x'], inputs['x_coord'])
    block_inputs = []
    final_inputs = []
    handles = [block['attention'].register_forward_pre_hook(
        lambda module, args: block_inputs.append(args[0].clone()),
    ) for block in model.patch_blocks]
    handles.append(model.final_layer.register_forward_pre_hook(
        lambda module, args: final_inputs.append(args[0].clone()),
    ))
    model(**inputs)
    for handle in handles:
        handle.remove()
    assert len(block_inputs) == depth
    for index, tokens in enumerate(block_inputs):
        torch.testing.assert_close(tokens, x_tokens + (index + 1) * coord_tokens)
    torch.testing.assert_close(final_inputs[0], x_tokens + depth * coord_tokens)


def test_self_attention_masks():
    """Boolean and additive masks agree and prevent reads from other patches."""
    model = make_model().eval()
    inputs = make_inputs()
    allowed = torch.eye(6, dtype=torch.bool)
    additive = torch.zeros(6, 6).masked_fill(~allowed, float('-inf'))
    expected = model(**inputs, mask=allowed)
    torch.testing.assert_close(model(**inputs, mask=additive), expected)
    changed_x = inputs['x'].detach().clone()
    changed_coords = inputs['x_coord'].detach().clone()
    changed_x[..., :2, :2] += 10
    changed_coords[..., :2, :2] += 3
    actual = model(**{**inputs, 'x': changed_x, 'x_coord': changed_coords}, mask=allowed)
    torch.testing.assert_close(actual[..., 2:, :], expected[..., 2:, :])
    torch.testing.assert_close(actual[..., :2, 2:], expected[..., :2, 2:])
    assert not torch.allclose(actual[..., :2, :2], expected[..., :2, :2])


def test_initialization_and_checkpoint(tmp_path):
    """Check initialization, config persistence, and nonzero output round-trip.

    Args:
        tmp_path: Temporary directory supplied by pytest for checkpoints.
    """
    model = ACDiT2DModel(in_channels=1, hidden_size=16, num_groups=2,
                         depth=3, patch_size=2, num_classes=2)
    inputs = make_inputs()
    assert torch.count_nonzero(model(**inputs)) == 0
    model = make_model()
    expected = model(**inputs)
    model.save_pretrained(tmp_path)
    restored = ACDiT2DModel.from_pretrained(tmp_path, low_cpu_mem_usage=False)
    torch.testing.assert_close(restored(**inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize('upcast', [False, True])
def test_cpu_autocast(upcast):
    """Check mixed-precision self-attention and coordinate gradients are finite.

    Args:
        upcast: Enable explicit float32 attention inside CPU autocast.
    """
    model = make_model(upcast_attention=upcast)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(**make_inputs(batch=1))
        loss = output.float().square().mean()
    assert torch.isfinite(output).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_initialization_can_train_coordinate_branch():
    """Verify ordinary optimizer steps unlock encoder and coordinate gradients."""
    torch.manual_seed(23)
    model = ACDiT2DModel(
        in_channels=1, hidden_size=16, num_groups=2, depth=3,
        patch_size=2, num_classes=2,
    )
    inputs = make_inputs(batch=1)
    target = torch.randn_like(inputs['x'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(**inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert model.x_patch_embedder.proj.weight.grad.abs().sum() > 0
    assert model.coords_patch_embedder.proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('width,heads,ratio', [(16, 2, 2.0), (32, 4, 4.0)])
@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'additive'])
def test_split_block_matches_original(width, heads, ratio, mask_kind):
    """Compare active outputs, input/condition gradients and parameter gradients.

    Args:
        width: Token width D.
        heads: Number of attention heads.
        ratio: MLP expansion ratio.
        mask_kind: Unmasked, boolean or additive self-attention mask.
    """
    torch.manual_seed(37)
    split = torch.nn.ModuleDict({
        'attention': ACDiTSelfAttnBlock(width, heads),
        'mlp': ACDiTMLPBlock(width, ratio),
    })
    original = AugmentedDiTBlock(width, heads, mlp_ratio=ratio)
    copy_split_block_to_original(split, original)
    assert sum(p.numel() for p in split.parameters()) == sum(
        p.numel() for p in original.parameters()
    )
    x = torch.randn(2, 6, width, requires_grad=True)
    c = torch.randn(2, 1, width, requires_grad=True)
    pos = precompute_freqs_cis_2d(width // heads, 2, 3)
    mask = None
    if mask_kind != 'none':
        mask = torch.ones(6, 6, dtype=torch.bool).tril()
        if mask_kind == 'additive':
            mask = torch.zeros(6, 6).masked_fill(~mask, float('-inf'))
    actual = split['mlp'](split['attention'](x, c, pos, mask), c)
    expected = original(x, c, pos, mask)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().mean(), (x, c), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().mean(), (x, c), retain_graph=True)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for split_module, original_module in [
        (split['attention'].norm, original.norm1),
        (split['attention'].attn, original.attn),
        (split['mlp'].norm, original.norm2),
        (split['mlp'].mlp, original.mlp),
    ]:
        for actual_param, expected_param in zip(split_module.parameters(), original_module.parameters()):
            torch.testing.assert_close(actual_param.grad, expected_param.grad)
    for name in ['weight', 'bias']:
        combined_grad = torch.cat([
            getattr(split['attention'].adaLN_modulation[0], name).grad,
            getattr(split['mlp'].adaLN_modulation[0], name).grad,
        ])
        torch.testing.assert_close(combined_grad, getattr(original.adaLN_modulation[0], name).grad)


@pytest.mark.parametrize('batch,width,heads,ref_grid', [
    (1, 16, 2, (1, 2)), (2, 32, 4, (3, 3)),
])
@pytest.mark.parametrize('upcast', [False, True])
def test_cross_attention_shapes_and_gradients(batch, width, heads, ref_grid, upcast):
    """Exercise unequal token grids and gradients to input, reference and condition.

    Args:
        batch: Batch size.
        width: Token and condition width D.
        heads: Attention head count.
        ref_grid: Reference patch grid (height, width), different from (2, 3).
        upcast: Compute attention in float32 when True.
    """
    torch.manual_seed(41)
    block = ACDiTCrossAttnBlock(width, heads, upcast_attention=upcast)
    x = torch.randn(batch, 6, width, requires_grad=True)
    ref = torch.randn(batch, ref_grid[0] * ref_grid[1], width, requires_grad=True)
    c = torch.randn(batch, 1, width, requires_grad=True)
    pos = precompute_freqs_cis_2d(width // heads, 2, 3)
    ref_pos = precompute_freqs_cis_2d(width // heads, *ref_grid)
    output = block(x, ref, c, pos, ref_pos)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    for tensor in [x, ref, c, *block.parameters()]:
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert not torch.allclose(block(x, ref, c, pos, ref_pos.flip(0)), output)


@pytest.mark.parametrize('upcast', [False, True])
def test_cross_attention_mask_and_reference_permutation(upcast):
    """Check masked reference exclusion and paired token/position permutation.

    Args:
        upcast: Compute attention in float32 when True.
    """
    torch.manual_seed(43)
    block = ACDiTCrossAttnBlock(16, 2, upcast_attention=upcast)
    x, ref, c = torch.randn(2, 6, 16), torch.randn(2, 8, 16), torch.randn(2, 1, 16)
    pos = precompute_freqs_cis_2d(8, 2, 3)
    ref_pos = precompute_freqs_cis_2d(8, 2, 4)
    mask = torch.ones(2, 1, 6, 8, dtype=torch.bool)
    mask[..., -2:] = False
    additive = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float('-inf'))
    expected = block(x, ref, c, pos, ref_pos, mask)
    torch.testing.assert_close(block(x, ref, c, pos, ref_pos, additive), expected)
    changed = ref.clone()
    changed[:, -2:] = torch.randn_like(changed[:, -2:]) * 10
    torch.testing.assert_close(block(x, changed, c, pos, ref_pos, mask), expected)
    order = torch.randperm(8)
    torch.testing.assert_close(
        block(x, ref[:, order], c, pos, ref_pos[order], mask[..., order]), expected,
    )


@pytest.mark.parametrize('upcast', [False, True])
def test_cross_attention_reduces_to_self_attention(upcast):
    """Match self-attention when references equal inputs and shift/scale are zero.

    Args:
        upcast: Compute attention in float32 when True.
    """
    torch.manual_seed(47)
    original = ACDiTSelfAttnBlock(16, 2, upcast_attention=upcast)
    cross = ACDiTCrossAttnBlock(16, 2, upcast_attention=upcast)
    with torch.no_grad():
        original.adaLN_modulation[0].weight.zero_()
        original.adaLN_modulation[0].bias.zero_()
        original.adaLN_modulation[0].bias[32:].fill_(1)
        for projection, weight in zip(
            [cross.q_proj, cross.k_proj, cross.v_proj], original.attn.qkv.weight.chunk(3),
        ):
            projection.weight.copy_(weight)
    for destination, source in [
        (cross.norm, original.norm), (cross.ref_norm, original.norm),
        (cross.q_norm, original.attn.q_norm), (cross.k_norm, original.attn.k_norm),
        (cross.proj, original.attn.proj), (cross.adaLN_modulation, original.adaLN_modulation),
    ]:
        destination.load_state_dict(source.state_dict())
    x = torch.randn(2, 6, 16, requires_grad=True)
    c = torch.randn(2, 1, 16)
    pos = precompute_freqs_cis_2d(8, 2, 3)
    actual, expected = cross(x, x, c, pos, pos), original(x, c, pos)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().mean(), x)[0]
    expected_grad = torch.autograd.grad(expected.square().mean(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('upcast', [False, True])
def test_cross_attention_cpu_autocast(upcast):
    """Check mixed precision on unequal grids with an additive attention mask.

    Args:
        upcast: Compute attention explicitly in float32 during CPU autocast.
    """
    torch.manual_seed(53)
    block = ACDiTCrossAttnBlock(16, 2, upcast_attention=upcast)
    x, ref, c = torch.randn(1, 6, 16), torch.randn(1, 8, 16), torch.randn(1, 1, 16)
    pos = precompute_freqs_cis_2d(8, 2, 3)
    ref_pos = precompute_freqs_cis_2d(8, 2, 4)
    mask = torch.zeros(6, 8)
    mask[:, -1] = float('-inf')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = block(x, ref, c, pos, ref_pos, mask)
        loss = output.float().square().mean()
    assert torch.isfinite(output).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters())


def test_cross_attention_zero_gate_is_identity():
    """Zero modulation preserves the input exactly through the gated residual."""
    block = ACDiTCrossAttnBlock(16, 2)
    torch.nn.init.zeros_(block.adaLN_modulation[0].weight)
    torch.nn.init.zeros_(block.adaLN_modulation[0].bias)
    x, ref, c = torch.randn(1, 6, 16), torch.randn(1, 8, 16), torch.randn(1, 1, 16)
    pos = precompute_freqs_cis_2d(8, 2, 3)
    ref_pos = precompute_freqs_cis_2d(8, 2, 4)
    torch.testing.assert_close(block(x, ref, c, pos, ref_pos), x, atol=0, rtol=0)


@pytest.mark.parametrize('patch,channels,depth', [(1, 1, 1), (2, 2, 3), (4, 1, 2)])
@pytest.mark.parametrize('upcast', [False, True])
def test_model_multiple_references(patch, channels, depth, upcast):
    """Verify reference influence and gradients through every cross-attention layer.

    Args:
        patch: Token patch side length.
        channels: Shared input/reference data channel count.
        depth: Number of self/cross/MLP layers.
        upcast: Compute attention in float32 when True.
    """
    model = make_model(use_cross_attention=True, patch_size=patch, in_channels=channels, depth=depth,
                       upcast_attention=upcast)
    inputs = make_inputs(patch, channels=channels)
    refs = [torch.randn(2, channels, patch, 2 * patch, requires_grad=True),
            torch.randn(2, channels, 3 * patch, patch, requires_grad=True)]
    coords = [torch.randn(2, 5, patch, 2 * patch, requires_grad=True),
              torch.randn(2, 5, 3 * patch, patch, requires_grad=True)]
    output = model(**inputs, r=refs, r_coord=coords)
    assert output.shape == (2, 2, 2 * patch, 3 * patch)
    assert not torch.allclose(output, model(**inputs))
    assert not torch.allclose(output, model(**inputs, r=[refs[0] + 2, refs[1]], r_coord=coords))
    torch.testing.assert_close(output, model(**inputs, r=refs[::-1], r_coord=coords[::-1]))
    torch.testing.assert_close(model(**inputs, r=refs[0], r_coord=coords[0]), model(**inputs, r=[refs[0]], r_coord=[coords[0]]))
    output.square().mean().backward()
    for tensor in [inputs['x'], inputs['x_coord'], *refs, *coords]:
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    for layer in model.patch_blocks:
        for parameter in layer['cross_attention'].parameters():
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('additive', [False, True])
def test_model_cross_mask_excludes_reference(additive):
    """Excluding one reference matches using the remaining reference alone.

    Args:
        additive: Use additive mask values instead of boolean allow values.
    """
    model = make_model(use_cross_attention=True)
    inputs = make_inputs()
    refs = [torch.randn(2, 1, 2, 4), torch.randn(2, 1, 6, 2)]
    coords = [torch.randn(2, 5, 2, 4), torch.randn(2, 5, 6, 2)]
    allowed = torch.ones(6, 5, dtype=torch.bool)
    allowed[:, 2:] = False
    mask = torch.zeros(6, 5).masked_fill(~allowed, float('-inf')) if additive else allowed
    self_mask = torch.eye(6, dtype=torch.bool)
    expected = model(**inputs, r=refs[0], r_coord=coords[0], mask=self_mask)
    actual = model(**inputs, r=refs, r_coord=coords, mask=self_mask, cross_mask=mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        model(**inputs, r=[refs[0], refs[1] + 10], r_coord=[coords[0], coords[1] + 10], mask=self_mask, cross_mask=mask),
        expected,
    )


def test_model_reference_checkpoint(tmp_path):
    """Verify a nonzero reference-conditioned prediction survives checkpoint reload.

    Args:
        tmp_path: Temporary checkpoint directory supplied by pytest.
    """
    model = make_model(use_cross_attention=True)
    inputs = make_inputs()
    refs = [torch.randn(2, 1, 2, 4), torch.randn(2, 1, 6, 2)]
    coords = [torch.randn(2, 5, 2, 4), torch.randn(2, 5, 6, 2)]
    expected = model(**inputs, r=refs, r_coord=coords)
    model.save_pretrained(tmp_path)
    restored = ACDiT2DModel.from_pretrained(tmp_path, low_cpu_mem_usage=False)
    assert restored.use_cross_attention is True
    assert restored.config.use_cross_attention is True
    torch.testing.assert_close(restored(**inputs, r=refs, r_coord=coords), expected, atol=0, rtol=0)


@pytest.mark.parametrize('upcast', [False, True])
def test_model_reference_cpu_autocast(upcast):
    """Verify mixed-precision forward/backward with multiple reference grids.

    Args:
        upcast: Compute attention explicitly in float32 during CPU autocast.
    """
    model = make_model(use_cross_attention=True, upcast_attention=upcast)
    refs = [torch.randn(1, 1, 2, 4), torch.randn(1, 1, 6, 2)]
    coords = [torch.randn(1, 5, 2, 4), torch.randn(1, 5, 6, 2)]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(**make_inputs(batch=1), r=refs, r_coord=coords)
        loss = output.float().square().mean()
    assert torch.isfinite(output).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_zero_initialization_can_train_cross_attention():
    """Optimizer steps unlock reference projections after zero-gated initialization."""
    torch.manual_seed(61)
    model = ACDiT2DModel(in_channels=1, hidden_size=16, num_groups=2,
                        depth=2, patch_size=2, num_classes=2, use_cross_attention=True)
    inputs = make_inputs(batch=1)
    ref = torch.randn(1, 1, 2, 4, requires_grad=True)
    ref_coord = torch.randn(1, 5, 2, 4, requires_grad=True)
    assert torch.count_nonzero(model(**inputs, r=ref, r_coord=ref_coord)) == 0
    for layer in model.patch_blocks:
        assert torch.count_nonzero(layer['cross_attention'].adaLN_modulation[0].weight) == 0
        assert torch.count_nonzero(layer['cross_attention'].adaLN_modulation[0].bias) == 0
    target = torch.randn_like(inputs['x'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        ref.grad = None
        ref_coord.grad = None
        loss = (model(**inputs, r=ref, r_coord=ref_coord) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(ref.grad).all() and ref.grad.abs().sum() > 0
    assert torch.isfinite(ref_coord.grad).all() and ref_coord.grad.abs().sum() > 0
    for layer in model.patch_blocks:
        assert layer['cross_attention'].k_proj.weight.grad.abs().sum() > 0
        assert layer['cross_attention'].v_proj.weight.grad.abs().sum() > 0


def test_model_cross_attention_disabled_ignores_references():
    """The disabled model has no cross parameters and ignores reference inputs."""
    model = make_model(use_cross_attention=False)
    inputs = make_inputs()
    r = torch.randn(2, 1, 2, 4, requires_grad=True)
    r_coord = torch.randn(2, 5, 2, 4, requires_grad=True)
    assert model.use_cross_attention is False
    assert all('cross_attention' not in block for block in model.patch_blocks)
    actual = model(**inputs, r=r, r_coord=r_coord)
    torch.testing.assert_close(actual, model(**inputs), atol=0, rtol=0)
    actual.square().mean().backward()
    assert r.grad is None and r_coord.grad is None


def test_model_reference_coordinates_affect_output():
    """Reference coordinates influence the prediction and receive gradients."""
    model = make_model(use_cross_attention=True)
    inputs = make_inputs()
    r = [torch.randn(2, 1, 2, 4), torch.randn(2, 1, 6, 2)]
    r_coord = [torch.randn(2, 5, 2, 4, requires_grad=True),
               torch.randn(2, 5, 6, 2, requires_grad=True)]
    actual = model(**inputs, r=r, r_coord=r_coord)
    changed = model(**inputs, r=r, r_coord=[r_coord[0] + 2, r_coord[1]])
    assert not torch.allclose(actual, changed)
    actual.square().mean().backward()
    for coords in r_coord:
        assert torch.isfinite(coords.grad).all() and coords.grad.abs().sum() > 0


def test_input_and_reference_share_embedding_parameters():
    """Identical input/reference pairs produce identical tokens in the same modules."""
    model = make_model(use_cross_attention=True)
    inputs = make_inputs()
    data_tokens, coordinate_tokens = [], []
    handles = [
        model.x_patch_embedder.register_forward_hook(
            lambda module, args, output: data_tokens.append(output.detach().clone()),
        ),
        model.coords_patch_embedder.register_forward_hook(
            lambda module, args, output: coordinate_tokens.append(output.detach().clone()),
        ),
    ]
    model(**inputs, r=inputs['x'], r_coord=inputs['x_coord'])
    for handle in handles:
        handle.remove()
    assert len(data_tokens) == len(coordinate_tokens) == 2
    torch.testing.assert_close(data_tokens[0], data_tokens[1], atol=0, rtol=0)
    torch.testing.assert_close(coordinate_tokens[0], coordinate_tokens[1], atol=0, rtol=0)
