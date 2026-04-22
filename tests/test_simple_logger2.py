import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import torch.distributed  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    dist = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
    )
    torch.distributed = dist
    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = dist


from core.logging import SimpleLogger2


class SimpleLogger2Test(unittest.TestCase):
    def test_writes_visible_demo_run_directory(self):
        demo_root = PROJECT_ROOT / "tests" / "logger2_demo_runs"
        logger = SimpleLogger2(
            root_dir=demo_root,
            run_name="visible_demo",
            console=False,
            logs={
                "train": [
                    "epoch",
                    "loss",
                    "velocity_loss",
                    "ssim_loss",
                    "lr",
                    "duration_sec",
                ],
                "valid": ["epoch", "loss", "psnr", "mae", "is_best"],
                "diagnostics": [
                    "epoch",
                    "scope",
                    "name",
                    "mean",
                    "std",
                    "nan_count",
                    "inf_count",
                ],
            },
        )

        demo_args = argparse.Namespace(
            model_name="unet",
            num_epochs=3,
            batch_size=8,
            learning_rate=1e-4,
            use_ssim_loss=True,
        )
        logger.log_system_info()
        logger.log_argparse_params(demo_args)
        logger.log_global_params(
            {
                "experiment_name": "logger2_visible_demo",
                "task": "seismic_reconstruction",
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "dataset": {
                    "train_path": "data/train_patches",
                    "valid_path": "data/valid_patches",
                    "patch_size": [32, 32],
                },
                "well_log_curves": {
                    "GR": "API",
                    "RHOB": "g/cm3",
                    "NPHI": "v/v",
                    "DT": "us/ft",
                },
            }
        )

        for epoch in range(1, 4):
            logger["train"].log_epoch(
                epoch,
                {
                    "loss": 0.5 / epoch,
                    "velocity_loss": 0.45 / epoch,
                    "ssim_loss": 0.05 / epoch,
                    "lr": 1e-4,
                    "duration_sec": 12.5 + epoch,
                },
            )

            logger["valid"].log_epoch(
                epoch,
                {
                    "loss": 0.62 / epoch,
                    "psnr": 22.0 + epoch,
                    "mae": 0.12 / epoch,
                    "is_best": epoch == 3,
                },
            )

            logger["diagnostics"].log(
                epoch=epoch,
                scope="grad",
                name="encoder.block1.weight",
                mean=1e-4 / epoch,
                std=2e-3 / epoch,
                nan_count=0,
                inf_count=0,
            )

            if epoch == 2:
                logger.log_event(
                    "checkpoint_saved",
                    epoch=epoch,
                    path=f"checkpoints/model_epoch_{epoch:05d}.pth",
                )

        logger.log_event(
            "best_model_updated",
            epoch=3,
            metric="valid/psnr",
            value=25.0,
            path="checkpoints/best.pth",
        )
        logger.close()

        print(f"\nSimpleLogger2 demo run: {logger.run_dir}")
        self.assertTrue(logger.run_dir.is_dir())
        self.assertTrue((logger.run_dir / "train.log").is_file())
        self.assertTrue((logger.run_dir / "valid.log").is_file())
        self.assertTrue((logger.run_dir / "diagnostics.log").is_file())
        self.assertTrue((logger.run_dir / "events.log").is_file())

    def test_creates_separate_run_files_and_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                root_dir=temp_dir,
                run_name="unit test",
                run_id="run001",
                console=False,
            )

            logger.log_global_params(
                {
                    "epochs": 2,
                    "batch_size": 4,
                    "curves": ["GR", "RHOB", "NPHI"],
                }
            )
            logger.log_system_info(include_git=False)
            logger.log_argparse_params(
                argparse.Namespace(
                    model_name="unet",
                    num_epochs=2,
                    batch_size=4,
                    learning_rate=1e-4,
                )
            )
            logger.log_train_epoch(1, {"loss": 0.12, "lr": 1e-4})
            logger.log_valid_epoch(1, {"loss": 0.2, "psnr": 30.5})
            logger.log_event("checkpoint_saved", path="model_epoch_00001.pth")
            logger.close()

            run_dir = Path(temp_dir) / "run001_unit_test"
            self.assertTrue(run_dir.is_dir())

            expected_files = {
                "train.log",
                "valid.log",
                "events.log",
            }
            actual_files = {path.name for path in run_dir.iterdir()}
            self.assertEqual(expected_files, actual_files)

            train_log = (run_dir / "train.log").read_text(encoding="utf-8")
            self.assertIn("[I] GLOBAL PARAMETERS", train_log)
            self.assertIn("[I] epochs: 2", train_log)
            self.assertIn("[I] batch_size: 4", train_log)
            self.assertIn("[I] curves: ['GR', 'RHOB', 'NPHI']", train_log)
            self.assertIn("[E] SYSTEM INFORMATION", train_log)
            self.assertIn("[E] python_version:", train_log)
            self.assertIn("[E] platform:", train_log)
            self.assertIn("[E] package_selection: deep_learning_defaults", train_log)
            self.assertIn("[E] package_torch:", train_log)
            self.assertIn("[E] package_numpy:", train_log)
            self.assertIn("[P] ARGPARSE PARAMETERS", train_log)
            self.assertIn("[P] model_name: unet", train_log)
            self.assertIn("[H] epoch loss lr", train_log)
            self.assertIn("[L] 1 0.12 0.0001", train_log)

            valid_log = (run_dir / "valid.log").read_text(encoding="utf-8")
            self.assertIn("[I] GLOBAL PARAMETERS", valid_log)
            self.assertIn("[H] epoch loss psnr", valid_log)
            self.assertIn("[L] 1 0.2 30.5", valid_log)

            events_log = (run_dir / "events.log").read_text(encoding="utf-8")
            self.assertIn("[I] event=run_started", events_log)
            self.assertIn("[I] event=global_params_written", events_log)
            self.assertIn("[I] event=system_info_written", events_log)
            self.assertIn("[I] event=argparse_params_written", events_log)
            self.assertIn("[I] event=checkpoint_saved", events_log)

    def test_configurable_log_channels_can_be_indexed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                root_dir=temp_dir,
                run_name="multi log",
                run_id="run001",
                console=False,
                logs={
                    "train": ["epoch", "loss", "lr"],
                    "valid": ["epoch", "loss", "psnr"],
                    "diagnostics": ["epoch", "scope", "name", "mean", "std"],
                },
            )

            self.assertIn("train", logger)
            self.assertIn("valid", logger)
            self.assertIn("diagnostics", logger)

            logger.log_global_params({"epochs": 1})
            logger.log_system_info(include_git=False, include_packages=False)
            logger.log_argparse_params(
                argparse.Namespace(model_name="unet", batch_size=8)
            )
            logger["train"].log_epoch(1, loss=0.12, lr=1e-4)
            logger["valid"].log_epoch(1, loss=0.2, psnr=30.5)
            logger["diagnostics"].log(
                epoch=1,
                scope="grad",
                name="encoder.weight",
                mean=0.001,
                std=0.02,
            )
            logger.close()

            run_dir = Path(temp_dir) / "run001_multi_log"
            actual_files = {path.name for path in run_dir.iterdir()}
            self.assertEqual(
                {"train.log", "valid.log", "diagnostics.log", "events.log"},
                actual_files,
            )

            train_log = (run_dir / "train.log").read_text(encoding="utf-8")
            self.assertLess(
                train_log.index("[I] GLOBAL PARAMETERS"),
                train_log.index("[H] epoch loss lr"),
            )
            self.assertIn("[L] 1 0.12 0.0001", train_log)

            diagnostics_log = (run_dir / "diagnostics.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("[H] epoch scope name mean std", diagnostics_log)
            self.assertIn("[L] 1 grad encoder.weight 0.001 0.02", diagnostics_log)

    def test_existing_run_directory_requires_overwrite_or_append(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SimpleLogger2(
                root_dir=temp_dir,
                run_name="same",
                run_id="run001",
                console=False,
            )
            logger.log_event("first_run")
            logger.close()

            with self.assertRaises(FileExistsError):
                SimpleLogger2(
                    root_dir=temp_dir,
                    run_name="same",
                    run_id="run001",
                    console=False,
                )

            logger = SimpleLogger2(
                root_dir=temp_dir,
                run_name="same",
                run_id="run001",
                append=True,
                console=False,
            )
            logger.log_event("second_run")
            logger.close()

            events_log = (
                Path(temp_dir) / "run001_same" / "events.log"
            ).read_text(
                encoding="utf-8"
            )
            self.assertIn("[I] event=first_run", events_log)
            self.assertIn("[I] event=second_run", events_log)


if __name__ == "__main__":
    unittest.main()
