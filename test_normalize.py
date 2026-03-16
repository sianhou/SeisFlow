import torch

from core.transforms.normalize import Normalize


def assert_in_range(x, low=-1.0, high=1.0, atol=1e-6):
    assert torch.all(x <= high + atol), f"max={float(x.max())} exceeds {high}"
    assert torch.all(x >= low - atol), f"min={float(x.min())} is below {low}"


def test_normalize_first_channel_minmax():
    transform = Normalize(mode="first_channel", method="minmax")
    x = torch.tensor(
        [
            [
                [[0.0, 2.0], [4.0, 6.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ]
        ],
        dtype=torch.float32,
    )

    y = transform(x)

    expected_ch0 = torch.tensor(
        [[[-1.0, -0.33333334], [0.33333334, 1.0]]],
        dtype=torch.float32,
    )

    assert torch.allclose(y[:, 0], expected_ch0, atol=1e-6)
    assert torch.allclose(y[:, 1], x[:, 1], atol=1e-6)
    assert_in_range(y[:, 0:1])


def test_normalize_perchannel_abs():
    transform = Normalize(mode="per_channel", method="abs")
    x = torch.tensor(
        [
            [
                [[-2.0, 0.0], [1.0, 2.0]],
                [[-10.0, 0.0], [5.0, 10.0]],
            ]
        ],
        dtype=torch.float32,
    )

    y = transform(x)

    expected = torch.tensor(
        [
            [
                [[-1.0, 0.0], [0.5, 1.0]],
                [[-1.0, 0.0], [0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(y, expected, atol=1e-6)
    assert_in_range(y)


def test_normalize_allchannel_rms():
    transform = Normalize(mode="all_channel", method="rms")
    x = torch.tensor(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        ],
        dtype=torch.float32,
    )

    y = transform(x)

    rms = torch.sqrt(torch.tensor((1.0 ** 2 * 4 + 2.0 ** 2 * 4) / 8.0, dtype=torch.float32))
    expected = (x / rms).clamp(-1.0, 1.0)

    assert torch.allclose(y, expected, atol=1e-6)
    assert_in_range(y)


def test_normalize_requires_rank_at_least_4():
    transform = Normalize(mode="per_channel", method="minmax")
    x = torch.ones(3, 4, 5, dtype=torch.float32)

    try:
        transform(x)
    except ValueError as exc:
        assert "ndim >= 4" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ndim < 4")


def test_normalize_invalid_mode():
    invalid_modes = ["bad_mode", "perchannel", "allchannel", "allchanell"]
    for mode in invalid_modes:
        try:
            Normalize(mode=mode, method="minmax")
        except ValueError as exc:
            assert "mode must be one of" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for invalid mode: {mode}")


def test_normalize_invalid_method():
    try:
        Normalize(mode="per_channel", method="bad_method")
    except ValueError as exc:
        assert "method must be one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid method")


def test_normalize_default_configuration():
    transform = Normalize()
    x = torch.tensor(
        [
            [
                [[-2.0, 0.0], [1.0, 2.0]],
                [[-10.0, 0.0], [5.0, 10.0]],
            ]
        ],
        dtype=torch.float32,
    )

    y = transform(x)
    expected = torch.tensor(
        [
            [
                [[-1.0, 0.0], [0.5, 1.0]],
                [[-1.0, 0.0], [0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(y, expected, atol=1e-6)
    assert_in_range(y)


def run_all_tests():
    test_normalize_first_channel_minmax()
    test_normalize_perchannel_abs()
    test_normalize_allchannel_rms()
    test_normalize_requires_rank_at_least_4()
    test_normalize_invalid_mode()
    test_normalize_invalid_method()
    test_normalize_default_configuration()
    print("All Normalize tests passed.")


if __name__ == "__main__":
    run_all_tests()
