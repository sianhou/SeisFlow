import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.transforms import Clip


def test_clip_shared_bounds_all_channels():
    x = torch.tensor(
        [
            [
                [[-3.0, -1.0], [1.0, 3.0]],
                [[-4.0, -2.0], [2.0, 4.0]],
            ]
        ]
    )

    y = Clip(vmin=-2.0, vmax=2.0, per_channel=False)(x)

    assert y.min() >= -2.0
    assert y.max() <= 2.0
    assert torch.allclose(y[0, 0], torch.tensor([[-2.0, -1.0], [1.0, 2.0]]))
    assert torch.allclose(y[0, 1], torch.tensor([[-2.0, -2.0], [2.0, 2.0]]))


def test_clip_per_channel_bounds():
    x = torch.tensor(
        [
            [
                [[-3.0, -1.0], [1.0, 3.0]],
                [[-4.0, -2.0], [2.0, 4.0]],
            ]
        ]
    )

    y = Clip(vmin=[-1.0, -3.0], vmax=[1.0, 3.0], per_channel=True)(x)

    assert torch.allclose(y[0, 0], torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    assert torch.allclose(y[0, 1], torch.tensor([[-3.0, -2.0], [2.0, 3.0]]))


def test_clip_rejects_inputs_below_3d():
    with pytest.raises(ValueError, match="ndim >= 3"):
        Clip(vmin=-1.0, vmax=1.0)(torch.tensor([1.0, 2.0]))


def test_clip_rejects_sequence_bounds_without_per_channel():
    x = torch.zeros(1, 2, 2)

    with pytest.raises(ValueError, match="per_channel=True"):
        Clip(vmin=[-1.0, -2.0], vmax=1.0, per_channel=False)(x)


if __name__ == "__main__":
    test_clip_shared_bounds_all_channels()
    test_clip_per_channel_bounds()
    test_clip_rejects_inputs_below_3d()
    test_clip_rejects_sequence_bounds_without_per_channel()

