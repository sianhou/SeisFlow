"""Compare V1 with V2 without cross attention, including migration pitfalls.

Run from the repository root with::

    python -m pytest tests/test_augmented_dit_v2_equivalence.py -q

All numeric comparisons require exact equality on the same CPU backend.
"""

import copy
import itertools
from contextlib import nullcontext

import pytest
import torch
from diffusers.training_utils import EMAModel

from models.cadit2d import CADiT2DModel
from models.pixeldit import AugmentedDiT2DModel
from models.wrapper import (
    AUGMENTED_DIT_2D_CONFIGS,
    AugmentedDiT2DWrapper,
    AugmentedDiT2DWrapperV2,
    build_augmented_dit_2d_wrapper,
    build_augmented_dit_2d_wrapper_v2,
)

BASE = dict(in_channels=3, out_channels=1, num_groups=2, hidden_size=16,
            depth=2, patch_size=2, num_classes=3, max_period=10)
MASKS = ("none", "all", "causal", "diagonal", "batch", "heads", "bias", "blocked")
GRID = list(itertools.product(
    (1, 3, 4, 15), (None, 1, 2), ((1, 4), (2, 8), (3, 24), (4, 64)),
    (1, 2, 4), (1, 2, 3, 4), (False, True),
))


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    """Use one CPU thread for small matrices; restore the prior setting after tests."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exact(left, right):
    """Assert finite, elementwise equality of two tensors with matching metadata."""
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def compare_state(left, right):
    """Compare ordered state tensors and parameter registration of two modules."""
    assert list(left.state_dict()) == list(right.state_dict())
    assert [n for n, _ in left.named_parameters()] == [n for n, _ in right.named_parameters()]
    for name, value in left.state_dict().items():
        exact(value, right.state_dict()[name])


def activate(model):
    """Give zero-initialized modulation and output layers nonzero weights in model."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "adaLN_modulation" in name or "final_layer.linear" in name:
                parameter.normal_(std=0.08)


def pair(config, seed=17, active=True, dtype=torch.float32):
    """Build equal V1/V2 models from config and seed, optionally activating gates.

    dtype selects parameter precision. Returns two CPU models after checking
    identical initial state and RNG consumption and strict loading both ways.
    """
    torch.manual_seed(seed)
    old = AugmentedDiT2DModel(**config)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    new = CADiT2DModel(**config, use_cross_attention=False)
    assert torch.equal(old_rng, torch.get_rng_state())
    compare_state(old, new)
    assert not any("cross" in n or "ref_" in n for n, _ in new.named_modules())
    if active:
        activate(old)
    new.load_state_dict(old.state_dict(), strict=True)
    old.load_state_dict(new.state_dict(), strict=True)
    return old.to(dtype=dtype), new.to(dtype=dtype)


def make_mask(kind, batch, heads, tokens, dtype):
    """Return a specified SDPA mask for [batch, heads, tokens, tokens] attention."""
    if kind == "none":
        return None
    if kind == "all":
        return torch.ones(tokens, tokens, dtype=torch.bool)
    if kind == "causal":
        return torch.ones(tokens, tokens, dtype=torch.bool).tril()
    if kind == "diagonal":
        return torch.eye(tokens, dtype=torch.bool)
    if kind in ("batch", "heads"):
        shape = (batch, 1 if kind == "batch" else heads, tokens, tokens)
        mask = torch.rand(shape) > 0.4
        mask.diagonal(dim1=-2, dim2=-1).fill_(True)
        return mask
    if kind == "bias":
        return torch.randn(tokens, tokens, dtype=dtype) * 0.5
    return torch.zeros(tokens, tokens, dtype=dtype).masked_fill(
        torch.ones(tokens, tokens, dtype=torch.bool).triu(1), float("-inf")
    )


def compare_forward_backward(old, new, x, t, y, mask=None, feature=0, amp=None):
    """Compare outputs, intermediate activations, input/time and all weight grads.

    x is BCHW, t has one value per batch item, y contains class indices. mask
    follows SDPA broadcasting; feature selects a block; amp selects CPU autocast
    dtype or disables autocast. Returns V1's detached prediction tensor.
    """
    runs = []
    module_names = ["patch_embedder", "t_embedder", "y_embedder", "final_layer"]
    module_names += [f"patch_blocks.{i}" for i in range(old.depth)]
    for model in (old, new):
        model.zero_grad(set_to_none=True)
        inputs = x.detach().clone().requires_grad_(True)
        times = t.detach().clone().requires_grad_(True)
        captured = []

        def capture(module, args, output):
            """Store a detached module output; module and args are hook inputs."""
            captured.append(output.detach().clone())

        handles = [model.get_submodule(n).register_forward_hook(capture) for n in module_names]
        context = torch.autocast("cpu", dtype=amp) if amp is not None else nullcontext()
        with context:
            result = model(inputs, times, y, mask=mask, return_patch_feature_at=feature)
            prediction, patch = result if feature is not None else (result, None)
            loss = (prediction.float() - 0.37).square().mean()
            if patch is not None:
                loss = loss + 0.13 * patch.float().square().mean()
        loss.backward()
        for handle in handles:
            handle.remove()
        runs.append((prediction.detach(), patch, captured, inputs.grad, times.grad, loss.detach()))
    for index in (0, 3, 4, 5):
        exact(runs[0][index], runs[1][index])
    if feature is not None:
        exact(runs[0][1], runs[1][1])
    for left, right in zip(runs[0][2], runs[1][2]):
        exact(left, right)
    for (name, left), (_, right) in zip(old.named_parameters(), new.named_parameters()):
        assert left.grad is not None, name
        assert right.grad is not None, name
        exact(left.grad, right.grad)
    return runs[0][0]


@pytest.mark.parametrize("case", range(len(GRID)))
def test_configuration_grid(case):
    """Check grid case across architecture, mask, batch, time, labels and mode."""
    channels, outputs, (groups, hidden), depth, patch, upcast = GRID[case]
    classes = (0, 1, 7, 1000)[(case // 7) % 4]
    config = dict(in_channels=channels, out_channels=outputs, num_groups=groups,
                  hidden_size=hidden, depth=depth, patch_size=patch,
                  num_classes=classes, max_period=(0.1, 1, 10, 37, 10000)[(case // 11) % 5],
                  upcast_attention=upcast)
    old, new = pair(config, seed=case % 11)
    train = (case // 13) % 2 == 0
    old.train(train)
    new.train(train)
    batch = (1, 2, 3)[(case // 5) % 3]
    rows, cols = ((1, 1), (1, 5), (5, 1), (2, 3), (4, 4), (3, 7))[(case // 17) % 6]
    x = torch.randn(batch, channels, rows * patch, cols * patch)
    if case % 3 == 0:
        x = x.transpose(-1, -2)
    elif case % 3 == 1:
        x = x.contiguous(memory_format=torch.channels_last)
    t = torch.linspace(-1, 2, batch) if case % 2 else torch.rand(batch)
    if case % 5 == 0:
        t = t[:, None]
    y = torch.arange(batch) % (classes + 1)
    y[-1] = classes
    mask = make_mask(MASKS[(case // 3) % len(MASKS)], batch, groups, rows * cols, x.dtype)
    feature = None if case % 4 == 0 else case % depth
    output = compare_forward_backward(old, new, x, t, y, mask, feature)
    assert output.shape == (batch, channels if outputs is None else outputs, *x.shape[-2:])
    assert output.abs().max() > 0


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("upcast", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mask_kind", MASKS)
def test_precision(dtype, upcast, training, mask_kind):
    """Check native dtype, upcast flag, training mode and mask_kind combinations."""
    old, new = pair({**BASE, "upcast_attention": upcast}, dtype=dtype)
    old.train(training)
    new.train(training)
    x = torch.randn(2, 3, 4, 6).to(dtype)
    mask = make_mask(mask_kind, 2, 2, 6, dtype)
    compare_forward_backward(old, new, x, torch.rand(2), torch.tensor([0, 3]), mask)


@pytest.mark.parametrize("amp", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("upcast", [False, True])
@pytest.mark.parametrize("mask_kind", MASKS)
def test_autocast(amp, upcast, mask_kind):
    """Check CPU autocast amp dtype, upcast flag and mask_kind with FP32 weights."""
    old, new = pair({**BASE, "upcast_attention": upcast})
    compare_forward_backward(old, new, torch.randn(2, 3, 4, 6), torch.rand(2),
                             torch.tensor([0, 3]), make_mask(mask_kind, 2, 2, 6, torch.float32), amp=amp)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize("active", [False, True])
def test_training_resume(optimizer_cls, active):
    """Check five training steps and optimizer-state migration for each initialization."""
    old, new = pair(BASE, active=active)
    options = dict(lr=0.003, momentum=0.9) if optimizer_cls is torch.optim.SGD else dict(lr=0.003)
    opt_old = optimizer_cls(old.parameters(), **options)
    opt_new = optimizer_cls(new.parameters(), **options)
    for step in range(5):
        compare_forward_backward(old, new, torch.randn(2, 3, 4, 6), torch.rand(2),
                                 torch.tensor([1, 3]), feature=None)
        opt_old.step()
        opt_new.step()
        compare_state(old, new)
        state_old, state_new = opt_old.state_dict(), opt_new.state_dict()
        assert state_old["param_groups"] == state_new["param_groups"]
        for key, values in state_old["state"].items():
            for field, value in values.items():
                if isinstance(value, torch.Tensor):
                    exact(value, state_new["state"][key][field])
                else:
                    assert value == state_new["state"][key][field]
        if step == 1:
            opt_new = optimizer_cls(new.parameters(), **options)
            opt_new.load_state_dict(copy.deepcopy(opt_old.state_dict()))


@pytest.mark.parametrize("upcast", [False, True])
@pytest.mark.parametrize("outputs", [None, 1, 5])
@pytest.mark.parametrize("safe", [False, True])
def test_checkpoint_both_directions(tmp_path, upcast, outputs, safe):
    """Check V1/V2 bidirectional config and checkpoint loading in both file formats."""
    old, new = pair({**BASE, "max_period": 37, "out_channels": outputs, "upcast_attention": upcast})
    for source, target_cls in ((old, CADiT2DModel), (new, AugmentedDiT2DModel)):
        path = tmp_path / type(source).__name__
        source.save_pretrained(path, safe_serialization=safe)
        restored, info = target_cls.from_pretrained(
            path, local_files_only=True, output_loading_info=True, use_safetensors=safe,
            low_cpu_mem_usage=False,
        )
        assert all(not info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        compare_state(source, restored)
        compare_forward_backward(source, restored, torch.randn(2, 3, 4, 6), torch.rand(2), torch.tensor([0, 3]))
        from_config = target_cls.from_config(source.config)
        from_config.load_state_dict(source.state_dict(), strict=True)
        compare_state(source, from_config)
        if target_cls is CADiT2DModel:
            assert restored.use_cross_attention is False
            assert from_config.use_cross_attention is False


@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("repa", [False, True])
@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_wrappers(concat, repa, labels, masked):
    """Check wrapper flags for conditioning, REPA, explicit labels and masks."""
    old, new = pair(BASE)
    old, new = AugmentedDiT2DWrapper(old), AugmentedDiT2DWrapperV2(new)
    if repa:
        old.configure_repa(2, 7)
        new.configure_repa(2, 7)
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 1 if concat else 3, 4, 6)
    extra = {}
    if concat:
        extra["concat_conditioning"] = torch.randn(2, 2, 4, 6)
    if labels:
        extra["label"] = torch.tensor([1, 3])
    if masked:
        extra["mask"] = torch.eye(6, dtype=torch.bool)
    t = torch.rand(2)
    results = []
    for wrapper in (old, new):
        result = wrapper(x, t, extra)
        tensors = result if repa else (result,)
        sum(value.square().mean() for value in tensors).backward()
        results.append(tensors)
    for left, right in zip(*results):
        exact(left, right)
    for left, right in zip(old.parameters(), new.parameters()):
        exact(left.grad, right.grad)


@pytest.mark.parametrize("name", AUGMENTED_DIT_2D_CONFIGS)
def test_full_preset_structure(name):
    """Compare full preset name on meta tensors, including builders and parameter order."""
    with torch.device("meta"):
        old = build_augmented_dit_2d_wrapper(model_arch=name)
        new = build_augmented_dit_2d_wrapper_v2(model_arch=name, use_cross_attention=False)
    left = [(n, p.shape, p.dtype, p.requires_grad) for n, p in old.named_parameters()]
    right = [(n, p.shape, p.dtype, p.requires_grad) for n, p in new.named_parameters()]
    assert left == right


@pytest.mark.parametrize("name", AUGMENTED_DIT_2D_CONFIGS)
def test_preset_width_numerics(name):
    """Exercise preset name at its real width/head count, using two blocks on CPU."""
    config = {**BASE, **AUGMENTED_DIT_2D_CONFIGS[name], "depth": 2}
    old, new = pair(config)
    compare_forward_backward(old, new, torch.randn(1, 3, 4, 6), torch.tensor([0.5]), torch.tensor([3]))


@pytest.mark.parametrize("name", ["Nano", "T"])
def test_full_small_presets(name):
    """Exercise complete Nano/T architecture name with nonzero weights and backward."""
    old, new = pair({**BASE, **AUGMENTED_DIT_2D_CONFIGS[name]})
    compare_forward_backward(old, new, torch.randn(1, 3, 4, 6), torch.tensor([0.5]), torch.tensor([3]))


def test_reference_ignored_and_positional_mask_not_compatible():
    """Demonstrate ignored r and silent positional-mask incompatibility of V2."""
    old, new = pair(BASE)
    x, t, y = torch.randn(2, 3, 4, 6), torch.rand(2), torch.tensor([0, 3])
    mask = torch.eye(6, dtype=torch.bool)
    with torch.no_grad():
        expected = old(x, t, y, mask)
        exact(expected, new(x, t, y, mask=mask))
        unmasked = new(x, t, y)
        exact(unmasked, new(x, t, y, mask))
        exact(unmasked, new(x, t, y, r=torch.randn(7)))
        difference = (expected - new(x, t, y, mask)).abs().max().item()
        print(f"positional mask silently ignored: max_abs_diff={difference:.9g}")
        assert difference > 1e-5
        old_result = old(x, t, y, None, 0)
        with pytest.raises((TypeError, RuntimeError)):
            new(x, t, y, None, 0)
        exact(old_result[1], new(x, t, y, return_patch_feature_at=0)[1])


def test_position_cache_and_each_feature_layer():
    """Check changing rectangles, same-token-count grids, cache reuse and every layer."""
    old, new = pair({**BASE, "depth": 5})
    for height, width in ((4, 6), (6, 4), (2, 12), (12, 2), (8, 8), (4, 6)):
        for index in range(5):
            compare_forward_backward(old, new, torch.randn(2, 3, height, width), torch.rand(2),
                                     torch.tensor([0, 3]), feature=index)
    assert old.precompute_pos.keys() == new.precompute_pos.keys()
    for key in old.precompute_pos:
        exact(old.precompute_pos[key], new.precompute_pos[key])


@pytest.mark.parametrize("change", [
    {"depth": 0}, {"depth": -1}, {"patch_size": 0}, {"patch_size": -2},
    {"hidden_size": 15}, {"hidden_size": 12}, {"max_period": 0}, {"max_period": -1},
])
def test_constructor_errors(change):
    """Check identical exception type/message for an invalid constructor change."""
    messages = []
    for cls in (AugmentedDiT2DModel, CADiT2DModel):
        with pytest.raises(ValueError) as error:
            cls(**{**BASE, **change})
        messages.append(str(error.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize("problem", ["rank", "channels", "height", "width", "negative_layer", "high_layer"])
def test_forward_errors(problem):
    """Check identical exception type/message for each specified forward problem."""
    old, new = pair(BASE)
    shape = {"rank": (2, 3, 4), "channels": (2, 4, 4, 6),
             "height": (2, 3, 5, 6), "width": (2, 3, 4, 7)}.get(problem, (2, 3, 4, 6))
    layer = {"negative_layer": -1, "high_layer": 2}.get(problem)
    messages = []
    for model in (old, new):
        with pytest.raises(ValueError) as error:
            model(torch.randn(shape), torch.rand(2), torch.tensor([0, 3]), return_patch_feature_at=layer)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize("depth", [1, 8, 12, 26])
@pytest.mark.parametrize("patch", [8, 16])
@pytest.mark.parametrize("grid", [(1, 1), (8, 16), (16, 16)])
@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_deep_large_patch_and_input_scale(depth, patch, grid, scale):
    """Check depth, patch size, token grid and input amplitude beyond the base grid."""
    old, new = pair({**BASE, "in_channels": 1, "hidden_size": 8,
                     "depth": depth, "patch_size": patch})
    x = torch.randn(1, 1, grid[0] * patch, grid[1] * patch) * scale
    compare_forward_backward(old, new, x, torch.tensor([1.0]), torch.tensor([3]), feature=depth - 1)


@pytest.mark.parametrize("upcast", [False, True])
@pytest.mark.parametrize("all_blocked", [False, True])
def test_mask_gradient_and_fully_blocked_rows(upcast, all_blocked):
    """Compare additive-mask gradients and SDPA all-blocked-row behavior for both flags."""
    old, new = pair({**BASE, "upcast_attention": upcast})
    x, t, y = torch.randn(2, 3, 4, 6), torch.rand(2), torch.tensor([0, 3])
    mask = torch.randn(6, 6)
    mask[0] = float("-inf")
    if all_blocked:
        mask.fill_(float("-inf"))
    outputs, gradients = [], []
    for model in (old, new):
        current = mask.clone().requires_grad_(True)
        output = model(x, t, y, mask=current)
        output.square().mean().backward()
        outputs.append(output)
        gradients.append(current.grad)
    exact(*outputs)
    exact(*gradients)


@pytest.mark.parametrize("time_kind", ["scalar", "integer", "column", "noncontiguous"])
def test_timestep_layouts(time_kind):
    """Check scalar, integer, column and strided timestep layout time_kind."""
    old, new = pair(BASE)
    batch = 1 if time_kind == "scalar" else 2
    t = {"scalar": torch.tensor(0.5), "integer": torch.tensor([0, 999]),
         "column": torch.tensor([[0.2], [0.7]]),
         "noncontiguous": torch.tensor([0.1, 0.2, 0.3, 0.4])[::2]}[time_kind]
    x, y = torch.randn(batch, 3, 4, 6), torch.zeros(batch, 1, dtype=torch.long)
    exact(old(x, t, y), new(x, t, y))


@pytest.mark.parametrize("source_v2", [False, True])
@pytest.mark.parametrize("use_ema", [False, True])
def test_wrapper_checkpoint_ema_and_optimizer(tmp_path, source_v2, use_ema):
    """Migrate saved wrapper, optimizer and optional EMA across versions in both directions."""
    old, new = pair(BASE)
    source_cls = AugmentedDiT2DWrapperV2 if source_v2 else AugmentedDiT2DWrapper
    target_cls = AugmentedDiT2DWrapper if source_v2 else AugmentedDiT2DWrapperV2
    source = source_cls(new if source_v2 else old)
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.001)
    ema = EMAModel(source.model.parameters(), decay=0.9,
                   model_cls=type(source.model), model_config=source.model.config)
    x, t = torch.randn(2, 3, 4, 6), torch.rand(2)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        source(x, t).square().mean().backward()
        optimizer.step()
        ema.step(source.model.parameters())
    source.save_pretrained(tmp_path, optimizer=optimizer, epoch=3, ema=ema)
    restored, epoch, training_state = target_cls.from_pretrained(
        tmp_path, return_training_state=True, use_ema=use_ema,
        low_cpu_mem_usage=False,
    )
    assert epoch == 3
    if use_ema:
        ema.copy_to(source.model.parameters())
    compare_state(source, restored)
    exact(source(x, t), restored(x, t))
    if not use_ema:
        resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
        resumed_optimizer.load_state_dict(training_state["optimizer"])
        for wrapper, opt in ((source, optimizer), (restored, resumed_optimizer)):
            opt.zero_grad(set_to_none=True)
            wrapper(x, t).square().mean().backward()
            opt.step()
        compare_state(source, restored)


def test_builder_positional_device_not_compatible():
    """Demonstrate that V2's inserted flag captures V1's positional device argument."""
    args = ("T", 3, 1, 2, 16, 2, 2, 3, 10, False, "cpu")
    old = build_augmented_dit_2d_wrapper(*args)
    new = build_augmented_dit_2d_wrapper_v2(*args)
    assert old.model.in_channels == new.model.in_channels
    assert new.model.use_cross_attention is True
    with pytest.raises(ValueError, match="reference input"):
        new(torch.randn(2, 3, 4, 6), torch.rand(2))
    fixed = build_augmented_dit_2d_wrapper_v2(
        **BASE, use_cross_attention=False, device="cpu",
    )
    assert fixed.model.use_cross_attention is False
