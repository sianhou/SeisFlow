import argparse
import logging
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


class SimpleLogger2Test(unittest.TestCase):
    def test_writes_single_visible_demo_log(self):
        demo_root = PROJECT_ROOT / "tests" / "logger2_demo_runs"
        logger = SimpleLogger2(
            output_dir=demo_root,
            log_id="visible_demo",
            overwrite=True,
            console=False,
            logs=["epoch", "loss", "psnr", "mae", "is_best"],
        )

        logger.log_system_info(include_git=False, include_packages=False)
        logger.log_argparse_params(
            argparse.Namespace(model_name="seismic_vae", num_epochs=3)
        )
        logger.log_global_params({"experiment_name": "logger2_visible_demo"})

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

        logger.close()

        print(f"\nSimpleLogger2 demo run: {logger.run_dir}")
        self.assertTrue(logger.run_dir.is_dir())
        self.assertTrue((logger.run_dir / "log.txt").is_file())

    def test_single_log_file_records_info_events_and_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                output_dir=temp_dir,
                log_id="run001",
                console=False,
                logs=["epoch", "loss", "lr"],
            )

            logger.log_global_params({"epochs": 2, "batch_size": 4})
            logger.log_system_info(include_git=False)
            logger.log_argparse_params(
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
            self.assertIn("\n\n[I] SYSTEM INFORMATION", log_text)
            self.assertIn("\n\n[I] ARGPARSE PARAMETERS", log_text)
            self.assertIn("\n\n[H] log_index epoch loss lr timestamp", log_text)
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
            logger.log_global_params({"ignored": "metadata"})
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

    def test_existing_run_directory_requires_overwrite_or_append(self):
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
