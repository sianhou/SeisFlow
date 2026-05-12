import sys
from pathlib import Path

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.transforms.normalize import AbsNormalize


def test_abs_normalize_3d_treats_first_two_dims_as_batch_channel():
    x = torch.tensor(
        [
            [[-2.0, 1.0], [4.0, 2.0]],
            [[-3.0, 1.0], [6.0, 2.0]],
        ]
    )

    y, scale = AbsNormalize(per_channel=True).run(x)
    expected_scale = torch.tensor([[[2.0], [4.0]], [[3.0], [6.0]]])
    assert torch.allclose(scale, expected_scale)
    assert torch.allclose(y * scale, x)


def test_abs_normalize_per_channel_returns_broadcastable_scale():
    x = torch.tensor(
        [
            [
                [[-2.0, 1.0], [0.5, 2.0]],
                [[-4.0, 1.0], [2.0, 3.0]],
            ]
        ]
    )

    y, scale = AbsNormalize(per_channel=True).run(x)
    expected_scale = torch.tensor([[[[2.0]], [[4.0]]]])
    assert torch.allclose(scale, expected_scale)
    assert torch.allclose(y * scale, x)


def test_abs_normalize_shared_channel_scale():
    x = torch.tensor(
        [
            [
                [[-2.0, 1.0], [0.5, 2.0]],
                [[-4.0, 1.0], [2.0, 3.0]],
            ]
        ]
    )
    transform = AbsNormalize(per_channel=False)
    y, scale = transform.run(x)
    assert torch.allclose(scale, torch.tensor([[[[4.0]]]]))
    assert torch.allclose(transform(x), y)
    assert torch.allclose(y * scale, x)


def test_abs_normalize_rejects_inputs_below_3d():
    transform = AbsNormalize()

    with pytest.raises(ValueError, match="ndim >= 3"):
        transform.run(torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    test_abs_normalize_3d_treats_first_two_dims_as_batch_channel()
    test_abs_normalize_per_channel_returns_broadcastable_scale()
    test_abs_normalize_shared_channel_scale()
    test_abs_normalize_rejects_inputs_below_3d()
