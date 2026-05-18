import argparse
import logging
import multiprocessing as mp
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


torch = types.ModuleType("torch")
dist = types.SimpleNamespace(
    is_available=lambda: False,
    is_initialized=lambda: False,
    get_rank=lambda: 0,
)
torch.distributed = dist
sys.modules["torch"] = torch
sys.modules["torch.distributed"] = dist


from core.logging import LogReader, SimpleLogger2
from core.logging.read_log import main as read_log_main


DEMO_ROOT = PROJECT_ROOT / "tests" / "logger2_demo_runs"


def _write_rank_log(output_dir, log_id, rank):
    logger = SimpleLogger2(
        output_dir=output_dir,
        log_id=log_id,
        append=True,
        console=False,
        log_file=f"log_rank{rank}.txt",
        logs=["rank", "step", "loss"],
    )
    logger.log_event("rank_started", rank=rank)
    logger.log_train(rank=rank, step=1, loss=0.1 * (rank + 1))
    logger.log_event("rank_finished", rank=rank)
    logger.close()


class SimpleLogger2Test(unittest.TestCase):
    def test_01_single_process_log_file(self):
        log_id = "single_process"
        shutil.rmtree(DEMO_ROOT / log_id, ignore_errors=True)
        logger = SimpleLogger2(
            output_dir=DEMO_ROOT,
            log_id=log_id,
            console=False,
            logs=["epoch", "loss", "psnr", "mae", "is_best"],
        )

        logger.log_system_info(include_git=False, include_packages=False)
        logger.log_info_block(
            "ARGPARSE PARAMETERS",
            argparse.Namespace(model_name="seismic_vae", num_epochs=3)
        )
        logger.log_info_block(
            "GLOBAL PARAMETERS",
            {"experiment_name": "logger2_single_process_demo"},
        )

        for epoch in range(1, 4):
            logger.log_train(
                epoch=epoch,
                loss=0.5 / epoch,
                psnr="",
                mae="",
                is_best=False,
            )
            logger.log_valid(
                epoch=epoch,
                loss=0.62 / epoch,
                psnr=22.0 + epoch,
                mae=0.12 / epoch,
                is_best=epoch == 3,
            )
            if epoch == 2:
                logger.log_event("checkpoint_saved", epoch=epoch)

        logger.log_event("single_process_finished")
        logger.close()

        log_path = DEMO_ROOT / log_id / "log.txt"
        self.assertTrue(log_path.is_file())
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("[I] info | event=run_started", log_text)
        self.assertIn("[I] SYSTEM INFORMATION", log_text)
        self.assertIn("[I] ARGPARSE PARAMETERS", log_text)
        self.assertIn("[I] GLOBAL PARAMETERS", log_text)
        self.assertIn("[I] info | event=checkpoint_saved | epoch=2", log_text)
        self.assertIn("[I] info | event=single_process_finished", log_text)
        self.assertIn("[H] log_index epoch loss psnr mae is_best timestamp", log_text)
        self.assertRegex(log_text, r"\[T\]\s+0\s+1\s+0\.5")
        self.assertRegex(log_text, r"\[V\]\s+1\s+1\s+0\.62\s+23")

    def test_02_spawn_four_processes_write_rank_logs(self):
        log_id = "spawn_four_ranks"
        run_dir = DEMO_ROOT / log_id
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        context = mp.get_context("spawn")
        processes = [
            context.Process(
                target=_write_rank_log,
                args=(str(DEMO_ROOT), log_id, rank),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)

        self.assertEqual(
            {f"log_rank{rank}.txt" for rank in range(4)},
            {path.name for path in run_dir.iterdir()},
        )
        for rank in range(4):
            log_text = (run_dir / f"log_rank{rank}.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn(f"[I] info | event=rank_started | rank={rank}", log_text)
            self.assertIn(f"[I] info | event=rank_finished | rank={rank}", log_text)
            self.assertIn("[H] log_index rank step loss timestamp", log_text)
            self.assertRegex(
                log_text,
                rf"\[T\]\s+0\s+{rank}\s+1\s+{0.1 * (rank + 1):.6g}",
            )

    def test_single_log_file_records_info_events_and_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                output_dir=temp_dir,
                log_id="run001",
                console=False,
                logs=["epoch", "loss", "lr"],
            )

            logger.log_info_block("GLOBAL PARAMETERS", {"epochs": 2, "batch_size": 4})
            logger.log_system_info(include_git=False)
            logger.log_info_block(
                "ARGPARSE PARAMETERS",
                argparse.Namespace(model_name="vae", learning_rate=1e-4)
            )
            logger.log_train(epoch=1, loss=0.12, lr=1e-4)
            logger.log_valid(epoch=1, loss=0.2, lr="")
            logger.log_event("checkpoint_saved", epoch=1)
            logger.log_event("nonfinite_grad", level=logging.WARNING, epoch=1)
            logger.close()

            run_dir = Path(temp_dir) / "run001"
            self.assertEqual({"log.txt"}, {path.name for path in run_dir.iterdir()})

            log_text = (run_dir / "log.txt").read_text(encoding="utf-8")
            self.assertIn("[I] GLOBAL PARAMETERS", log_text)
            self.assertIn("[I] SYSTEM INFORMATION", log_text)
            self.assertIn("[I] ARGPARSE PARAMETERS", log_text)
            self.assertIn("[I] info | event=run_started", log_text)
            self.assertIn("[I] info | event=checkpoint_saved | epoch=1", log_text)
            self.assertIn("[I] warning | event=nonfinite_grad | epoch=1", log_text)
            self.assertIn("[I]\n[I]\n[I] SYSTEM INFORMATION", log_text)
            self.assertIn("[I]\n[I]\n[I] ARGPARSE PARAMETERS", log_text)
            self.assertIn("[I]\n[I]\n[H] log_index epoch loss lr timestamp", log_text)
            self.assertRegex(log_text, r"\[T\]\s+0\s+1\s+0\.12\s+0\.0001")
            self.assertRegex(log_text, r"\[V\]\s+1\s+1\s+0\.2")

    def test_log_rows_have_counter_and_fixed_width_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                output_dir=temp_dir,
                log_id="run001",
                console=False,
                logs=["epoch", "loss"],
                log_value_width=6,
            )
            logger.log_train(epoch=1, loss=0.12)
            logger.log_train(epoch=2, loss=0.11)
            logger.close()

            log_text = (Path(temp_dir) / "run001" / "log.txt").read_text(
                encoding="utf-8"
            )
            table_lines = [
                line
                for line in log_text.splitlines()
                if line.startswith("[H] ")
                or line.startswith("[T] ")
                or line.startswith("[V] ")
            ]
            self.assertEqual("[H] log_index epoch loss timestamp", table_lines[0])
            self.assertRegex(
                table_lines[1],
                r"^\[T\]\s+0\s+1\s+0\.12\s+\d+$",
            )
            self.assertRegex(
                table_lines[2],
                r"^\[T\]\s+1\s+2\s+0\.11\s+\d+$",
            )

    def test_read_log_filters_by_line_prefix_and_exports_column_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                output_dir=temp_dir,
                log_id="run001",
                console=False,
                logs=["epoch", "loss", "is_best"],
            )
            logger.log_info_block("GLOBAL PARAMETERS", {"ignored": "metadata"})
            logger.log_train(epoch=1, loss=0.12, is_best=False)
            logger.log_valid(epoch=1, loss=0.2, is_best=False)
            logger.log_train(epoch=2, loss=0.11, is_best=True)
            logger.close()

            run_dir = Path(temp_dir) / "run001"
            reader = LogReader(run_dir, channel="T")
            self.assertEqual(
                ["log_index", "epoch", "loss", "is_best", "timestamp"],
                reader.columns,
            )
            self.assertEqual([0, 2], reader["log_index"])
            self.assertEqual([1, 2], reader["epoch"])
            self.assertEqual([0.12, 0.11], reader["loss"])
            self.assertEqual([False, True], reader["is_best"])

            output_dir = reader.export_columns()
            self.assertEqual(
                "0 0\n2 2\n",
                (output_dir / "log_index.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                "0 0.12\n2 0.11\n",
                (output_dir / "loss.txt").read_text(encoding="utf-8"),
            )

            cli_output_dir = run_dir / "cli_columns"
            result = read_log_main(
                [
                    "--path",
                    str(run_dir / "log.txt"),
                    "--channel",
                    "T",
                    "--output_dir",
                    str(cli_output_dir),
                ]
            )
            self.assertEqual(cli_output_dir, result)
            self.assertEqual(
                "0 0\n2 2\n",
                (cli_output_dir / "log_index.txt").read_text(encoding="utf-8"),
            )

    def test_existing_log_file_requires_overwrite_or_append(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                output_dir=temp_dir,
                log_id="run001",
                console=False,
            )
            logger.log_event("first_run")
            logger.close()

            with self.assertRaises(FileExistsError):
                SimpleLogger2(
                    output_dir=temp_dir,
                    log_id="run001",
                    console=False,
                )

            logger = SimpleLogger2(
                output_dir=temp_dir,
                log_id="run001",
                append=True,
                console=False,
            )
            logger.log_event("second_run")
            logger.close()

            log_text = (Path(temp_dir) / "run001" / "log.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("[I] info | event=first_run", log_text)
            self.assertIn("[I] info | event=second_run", log_text)


if __name__ == "__main__":
    unittest.main()
