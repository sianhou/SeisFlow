"""Check ordered training-shot references and their saved patch alignment."""

import numpy as np
import pytest
import segyio

import BuildShotDataset2 as builder


@pytest.fixture
def shot_file(tmp_path):
    """Return a six-shot SEG-Y fixture with shuffled records and nonmonotonic XY.

    Args:
        tmp_path: Pytest temporary directory for the source file.
    """
    path = tmp_path / "shots.sgy"
    spec = segyio.spec()
    spec.format = 5
    spec.samples = range(5)
    spec.tracecount = 18
    source_x = [0, 1000, 5, 900, 10, 800]
    with segyio.create(str(path), spec) as output:
        for record, shot in enumerate([5, 0, 3, 1, 4, 2]):
            for receiver in range(3):
                trace = record * 3 + receiver
                output.trace[trace] = (
                    np.arange(5, dtype=np.float32) + 10 * shot + receiver
                )
                output.header[trace] = {
                    segyio.TraceField.FieldRecord: 10 * (shot + 1),
                    segyio.TraceField.SourceX: source_x[shot],
                    segyio.TraceField.SourceY: 20,
                    segyio.TraceField.GroupX: source_x[shot] + receiver * 10,
                    segyio.TraceField.GroupY: 30,
                    segyio.TraceField.SourceGroupScalar: -10,
                    segyio.TraceField.CoordinateUnits: 1,
                }
    return path


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("valid_ratio", [0, 0.5])
@pytest.mark.parametrize("normalize_coords", [False, True])
def test_reference_outputs(shot_file, tmp_path, monkeypatch, count, valid_ratio, normalize_coords):
    """Verify counts, ordering, edge fallback, metadata and both coordinate modes.

    Args:
        shot_file: Synthetic SEG-Y with six sorted shot IDs.
        tmp_path: Pytest directory for output patches.
        monkeypatch: Fixture disabling unrelated presence-plot rendering.
        count: Requested number of reference slots.
        valid_ratio: Validation split fraction, including no validation.
        normalize_coords: Whether coordinates are normalized before copying.
    """
    monkeypatch.setattr(builder, "plot_shot_presence", lambda *args: None)
    root = tmp_path / "output"
    argv = [
        "--segy", str(shot_file), "--output_dir", str(root),
        "--gen-ref", str(count), "--patch_size", "2", "--overlap_size", "1",
        "--valid", str(valid_ratio), "--slice", "1", "5", "--normalize",
    ]
    if not normalize_coords:
        argv.append("--no_normalize_coords")
    builder.build_dataset(builder.build_parser().parse_args(argv))
    valid = {1, 3, 5} if valid_ratio else set()
    expected = (
        [[2, 4], [0, 2], [0, 4], [2, 4], [2, 0], [4, 2]]
        if valid_ratio else
        [[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [4, 3]]
    )
    expected_dirs = set(builder.build_output_dirs(str(root), count))
    assert {path.name for path in root.iterdir()} == expected_dirs
    for target in range(6):
        split = "valid" if target in valid else "train"
        selected_references = []
        for slot in range(count):
            reference = expected[target][slot]
            assert reference not in valid and reference != target
            tag = "ref" if slot == 0 else "ref2"
            target_name = f"patches_{target:04d}"
            ref_name = f"patches_{reference:04d}"
            for suffix in ("", "_dim"):
                actual = np.load(root / f"{split}_{tag}{suffix}" / f"{target_name}.npy")
                source = np.load(root / f"train{suffix}" / f"{ref_name}.npy")
                np.testing.assert_array_equal(actual, source)
            with np.load(root / f"{split}_{tag}_aux" / f"{target_name}.npz") as metadata:
                assert metadata["target_shot_id"] == 10 * (target + 1)
                assert metadata["reference_shot_id"] == 10 * (reference + 1)
                assert metadata["reference_shot_index"] == reference
                selected_references.append(int(metadata["reference_shot_index"]))
                np.testing.assert_allclose(
                    metadata["reference_distance"],
                    np.linalg.norm(metadata["target_source_xy"] - metadata["reference_source_xy"]),
                )
                for directory, name in [(f"{split}_aux", target_name), ("train_aux", ref_name)]:
                    with np.load(root / directory / f"{name}.npz") as source:
                        for key in ("positions", "original_shape", "global_scale",
                                    "global_coord_min", "global_coord_max"):
                            np.testing.assert_array_equal(metadata[key], source[key])
        assert len(set(selected_references)) == count


def test_reference_count_default():
    """Omitting --gen-ref disables reference generation."""
    assert builder.build_parser().parse_args(["--segy", "unused.sgy"]).gen_ref == 0


@pytest.mark.parametrize("arguments", [["--gen-ref"], ["--gen-ref", "3"]])
def test_reference_count_requires_supported_integer(arguments):
    """Reject a bare switch and unsupported counts.

    Args:
        arguments: Invalid reference-option CLI arguments.
    """
    with pytest.raises(SystemExit):
        builder.build_parser().parse_args(["--segy", "unused.sgy", *arguments])
