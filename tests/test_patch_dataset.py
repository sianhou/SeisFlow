import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import PatchDataset


def _write_patch_files(root, patches):
    npy_dir = root / "npy"
    npz_dir = root / "npz"
    npy_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    np.save(npy_dir / "patches_0000.npy", patches)
    np.savez(npz_dir / "patches_0000.npz", patches=patches)
    return npy_dir, npz_dir


def _read_with_dataloader(data_dir, npy, batch_size, num_workers, loops):
    dataset = PatchDataset(data_dir, npy=npy)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    batches = []
    start = time.perf_counter()
    for _ in range(loops):
        loop_batches = []
        for batch in dataloader:
            loop_batches.append(batch)
        batches.append(torch.cat(loop_batches, dim=0))
    elapsed = time.perf_counter() - start

    return batches, elapsed


def run_patch_dataset_io_check(
        num_patches=20,
        patch_size=256,
        loops=20,
        batch_size=4,
        num_workers=0,
        seed=0,
):
    rng = np.random.default_rng(seed)
    patches = rng.standard_normal(
        (num_patches, patch_size, patch_size),
        dtype=np.float32,
    )
    expected = torch.from_numpy(patches).unsqueeze(1)

    root = Path(tempfile.mkdtemp(prefix="patch_dataset_io_"))
    try:
        npy_dir, npz_dir = _write_patch_files(root, patches)

        npy_batches, npy_seconds = _read_with_dataloader(
            npy_dir,
            npy=True,
            batch_size=batch_size,
            num_workers=num_workers,
            loops=loops,
        )
        npz_batches, npz_seconds = _read_with_dataloader(
            npz_dir,
            npy=False,
            batch_size=batch_size,
            num_workers=num_workers,
            loops=loops,
        )

        for batch in npy_batches:
            assert torch.equal(batch, expected)
        for batch in npz_batches:
            assert torch.equal(batch, expected)

        return {
            "num_patches": num_patches,
            "patch_size": patch_size,
            "loops": loops,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "npy_seconds": npy_seconds,
            "npz_seconds": npz_seconds,
        }
    finally:
        shutil.rmtree(root)


def test_patch_dataset_reads_npy_and_npz_correctly():
    stats = run_patch_dataset_io_check()
    print(
        "PatchDataset IO benchmark: "
        f"npy={stats['npy_seconds']:.6f}s, "
        f"npz={stats['npz_seconds']:.6f}s, "
        f"loops={stats['loops']}, "
        f"num_patches={stats['num_patches']}, "
        f"patch_size={stats['patch_size']}"
    )


if __name__ == "__main__":
    result = run_patch_dataset_io_check(num_workers=0)
    print("PatchDataset IO benchmark")
    print(f"  num_patches: {result['num_patches']}")
    print(f"  patch_size: {result['patch_size']}")
    print(f"  loops: {result['loops']}")
    print(f"  batch_size: {result['batch_size']}")
    print(f"  num_workers: {result['num_workers']}")
    print(f"  npy_seconds: {result['npy_seconds']:.6f}")
    print(f"  npz_seconds: {result['npz_seconds']:.6f}")
