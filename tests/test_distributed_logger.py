"""Direct terminal demo for the distributed logger.

Run from the project root:
    python tests/test_distributed_logger.py
"""

import multiprocessing as mp
import os
import shutil
import socket
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Keep this demo independent from CUDA/PyTorch initialization. The logger only
# needs the small distributed API below for this simulated multi-node run.
torch = types.ModuleType("torch")
dist = types.ModuleType("torch.distributed")
dist.is_available = lambda: False
dist.is_initialized = lambda: False
dist.get_rank = lambda: 0
torch.distributed = dist
sys.modules["torch"] = torch
sys.modules["torch.distributed"] = dist


from core.logging.logger import build_dist_logger


LOG_ROOT = PROJECT_ROOT / "temp" / "distributed_logger_test_logs"
LOG_ID = "build_dist_logger_demo"
WORLD_SIZE = 2


def run_simulated_node(rank):
    """Run one simulated node and give it a unique hostname-based log file."""
    os.chdir(PROJECT_ROOT)
    node_name = f"test-node-{rank}"
    original_gethostname = socket.gethostname
    socket.gethostname = lambda: node_name
    try:
        args = types.SimpleNamespace(
            output_dir=str(LOG_ROOT),
            distributed=True,
            rank=rank,
            world_size=WORLD_SIZE,
            gpu=0,
            log_console=False,
            log_id=LOG_ID,
        )
        logger = build_dist_logger(
            args,
            log_node_info=True,
            job_file=__file__,
        )
        logger.log_event("simulated_node_finished", rank=rank)
        logger.close()
    finally:
        socket.gethostname = original_gethostname


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    run_dir = LOG_ROOT / LOG_ID
    shutil.rmtree(run_dir, ignore_errors=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    processes = [
        context.Process(target=run_simulated_node, args=(rank,))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        require(process.exitcode == 0, f"node process failed: {process.exitcode}")

    expected_files = {
        f"log_test-node-{rank}.txt" for rank in range(WORLD_SIZE)
    }
    actual_files = {path.name for path in run_dir.iterdir()}
    require(actual_files == expected_files, f"unexpected log files: {actual_files}")

    for rank in range(WORLD_SIZE):
        log_path = run_dir / f"log_test-node-{rank}.txt"
        log_text = log_path.read_text(encoding="utf-8")
        required_fragments = [
            "[I] RUNTIME ENVIRONMENT",
            "main_program: test_distributed_logger.py",
            "python_version:",
            "git_commit:",
            f"hostname: test-node-{rank}",
            f"rank: {rank}",
            f"event=simulated_node_finished | rank={rank}",
        ]
        for fragment in required_fragments:
            require(fragment in log_text, f"{fragment!r} missing from {log_path}")

    print("Distributed logger demo passed.")
    print(f"Logs written to: {run_dir}")
    for path in sorted(run_dir.glob("*.txt")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
