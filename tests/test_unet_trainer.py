"""Exercise the standalone entry point with real NPY data and checkpoints."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def test_unet_cli_train_resume_and_sample(tmp_path):
    seismic_dir = tmp_path / "seismic"
    coordinate_dir = tmp_path / "coordinates"
    seismic_dir.mkdir()
    coordinate_dir.mkdir()
    rng = np.random.default_rng(0)
    np.save(seismic_dir / "shot.npy", rng.normal(size=(2, 8, 12)).astype("float32"))
    np.save(coordinate_dir / "shot.npy", rng.uniform(-1, 1, (2, 2, 8, 12)).astype("float32"))

    root = Path(__file__).resolve().parents[1]
    script = root / "UNetSeisDimReconNeRF.py"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def invoke(*args):
        result = subprocess.run(
            [sys.executable, str(script), *map(str, args)],
            cwd=root, env=env, capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    common = [
        "--device", "cpu", "--num_workers", "0", "--batch_size", "2",
        "--input_dim_dir", coordinate_dir, "--nerf_bands", "1",
    ]
    train = [
        *common, "--input_dir", seismic_dir, "--model_arch", "Nano",
        "--max_period", "25", "--frequency_embedding_size", "64",
        "--save_every_epochs", "1", "--use_checkpoint",
    ]
    invoke(*train, "--num_epochs", "1", "--output_dir", tmp_path / "train")
    checkpoint = next((tmp_path / "train").rglob("checkpoint_epoch_00001"))
    assert (checkpoint / "ema" / "config.json").is_file()

    invoke(*train, "--num_epochs", "2", "--ckpt", checkpoint,
           "--output_dir", tmp_path / "resume")
    resumed = next((tmp_path / "resume").rglob("checkpoint_epoch_00002"))
    state = torch.load(resumed / "training_state.pth", weights_only=False)
    assert state["epoch"] == 2
    assert state["optimizer"]["state"]

    # Sampling restores architecture/time settings, without repeating those flags.
    invoke("sample", *common, "--ckpt", resumed, "--solver_step_size", "0.5",
           "--output_dir", tmp_path / "sample")
    output_path = next((tmp_path / "sample").rglob("shot.npy"))
    output = np.load(output_path)
    assert output.shape == (2, 1, 8, 12)
    assert np.isfinite(output).all()
