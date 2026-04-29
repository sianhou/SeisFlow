import logging
import os
import platform
import socket
import subprocess
import sys
import re
from datetime import datetime
from getpass import getuser
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch.distributed as dist


class SimpleLogger:
    """
    A minimal logger for PyTorch training.

    Features:
        - Outputs to both stdout and a log file
        - Automatically creates the log directory
        - Only logs from rank 0 in distributed training
        - Prevents duplicate handlers when re-instantiated

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_file (str): Name of the log file.
        level (int): Logging level (default: logging.INFO).
        overwrite (bool): If True, overwrite existing log file.
        append (bool): If True, append to existing log file. Ignored if overwrite=True.


    Example:
        logger = BaseLogger(log_dir="./logs", log_file="train.log")
        logger.info("Start training")
    """

    def __init__(self,
                 log_dir="./results",
                 log_file="train.log",
                 level=logging.INFO,
                 overwrite=False,
                 append=False):
        self.log_dir = log_dir
        self.log_file = log_file
        self.overwrite = overwrite
        self.append = append

        # Create log directory if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Determine file mode
        file_path = os.path.join(self.log_dir, self.log_file)
        if os.path.exists(file_path):
            if self.overwrite:
                mode = "w"
            elif self.append:
                mode = "a"
            else:
                raise FileExistsError(
                    f"Log file '{file_path}' already exists. "
                    "Set overwrite=True to overwrite or append=True to append."
                )
        else:
            mode = "w"

        self.logger = logging.getLogger("base_logger")
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent propagation to root logger

        # Avoid adding duplicate handlers
        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # stdout handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # file handler
        file_handler = logging.FileHandler(file_path, mode=mode)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _is_main_process(self):
        """
        Check whether the current process is the main process (rank 0).
        """
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def info(self, msg):
        if self._is_main_process():
            self.logger.info(msg)

    def warning(self, msg):
        if self._is_main_process():
            self.logger.warning(msg)

    def error(self, msg):
        if self._is_main_process():
            self.logger.error(msg)


class RunLoggerBase:
    """
    Base class for run-scoped loggers.

    A run-scoped logger creates one directory for one experiment run and keeps
    channel-specific log files inside that directory.
    """

    def __init__(
        self,
        root_dir: str = "./results",
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        level: int = logging.INFO,
        overwrite: bool = False,
        append: bool = False,
        console: bool = True,
    ):
        if overwrite and append:
            raise ValueError("overwrite and append cannot both be True.")

        self.root_dir = Path(root_dir)
        self.run_name = run_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.level = level
        self.overwrite = overwrite
        self.append = append
        self.console = console
        self._loggers: Dict[str, logging.Logger] = {}

        folder_name = self.run_id
        if run_name:
            folder_name = f"{self.run_id}_{self._safe_name(run_name)}"
        self.run_dir = self.root_dir / folder_name

        if self._is_main_process():
            self._prepare_run_dir()

    @staticmethod
    def _safe_name(value: str) -> str:
        value = value.strip()
        value = re.sub(r'[<>:"/\\|?*\s]+', "_", value)
        value = value.strip("._")
        return value or "run"

    def _prepare_run_dir(self):
        if self.run_dir.exists():
            has_contents = any(self.run_dir.iterdir())
            if has_contents and not self.overwrite and not self.append:
                raise FileExistsError(
                    f"Run directory '{self.run_dir}' already exists. "
                    "Set overwrite=True or append=True, or use a new run_id."
                )
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _file_mode(self) -> str:
        if self.append:
            return "a"
        return "w"

    def _is_main_process(self) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _build_logger(self, channel: str, file_name: str) -> logging.Logger:
        if channel in self._loggers:
            return self._loggers[channel]

        logger_name = (
            f"{self.__class__.__name__}.{self.run_id}.{channel}.{id(self)}"
        )
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        logger.propagate = False

        formatter = logging.Formatter(fmt="%(message)s")

        file_path = self.run_dir / file_name
        file_handler = logging.FileHandler(
            file_path,
            mode=self._file_mode(),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if self.console:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        self._loggers[channel] = logger
        return logger

    def _log(self, channel: str, level: int, message: str):
        if not self._is_main_process():
            return
        logger = self._loggers[channel]
        logger.log(level, message)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return str(value.item())
            except (TypeError, ValueError):
                pass
        if hasattr(value, "tolist"):
            try:
                return str(value.tolist())
            except (TypeError, ValueError):
                pass
        return str(value)

    def _format_record(self, message: Optional[str], fields: Mapping[str, Any]) -> str:
        parts = []
        if message:
            parts.append(str(message))
        for key, value in fields.items():
            parts.append(f"{key}={self._format_value(value)}")
        return " | ".join(parts)

    def close(self):
        for logger in self._loggers.values():
            for handler in list(logger.handlers):
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
        self._loggers.clear()


class SimpleLogChannel:
    """
    One LAS-like log channel inside SimpleLogger2.
    """

    def __init__(self, parent: "SimpleLogger2", name: str):
        self.parent = parent
        self.name = name

    def info(self, message: str):
        self.parent._write_info(self.name, message)

    def header(self, fields: Sequence[str]):
        self.parent._write_header(self.name, fields)

    def log(self, **fields: Any):
        self.parent._log_curve_row(self.name, None, fields)

    def log_epoch(
        self,
        epoch: int,
        metrics: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ):
        record = {"epoch": epoch}
        if metrics:
            record.update(metrics)
        record.update(fields)
        self.log(**record)


class SimpleLogger2(RunLoggerBase):
    """
    Run-scoped training logger with LAS-like train, valid, and event logs.

    By default, files created in each run directory:
        - train.log
        - valid.log
        - events.log

    Example:
        logger = SimpleLogger2(root_dir="./logs", run_name="unet")
        logger.log_global_params({"epochs": 100, "batch_size": 16})
        logger.log_train_epoch(1, {"loss": 0.23, "lr": 1e-4})
        logger.log_valid_epoch(1, {"loss": 0.31, "psnr": 24.7})
        logger["train"].log_epoch(2, {"loss": 0.19, "lr": 1e-4})
        logger.log_event("checkpoint_saved", path="model_epoch_00001.pth")
    """

    def __init__(
        self,
        root_dir: str = "./results",
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        level: int = logging.INFO,
        overwrite: bool = False,
        append: bool = False,
        console: bool = True,
        logs: Optional[Any] = None,
        event_log: str = "events",
        log_value_width: int = 10,
    ):
        if log_value_width < 1:
            raise ValueError("log_value_width must be >= 1.")

        super().__init__(
            root_dir=root_dir,
            run_name=run_name,
            run_id=run_id,
            level=level,
            overwrite=overwrite,
            append=append,
            console=console,
        )
        self._headers: Dict[str, List[str]] = {}
        self._log_counters: Dict[str, int] = {}
        self.event_log = event_log
        self.log_value_width = log_value_width
        self.log_names, self._configured_headers = self._normalize_logs(logs)
        self._channels: Dict[str, SimpleLogChannel] = {
            name: SimpleLogChannel(self, name) for name in self.log_names
        }

        if self._is_main_process():
            for name in self.log_names:
                self._build_logger(name, f"{name}.log")
            self._build_logger(self.event_log, f"{self.event_log}.log")
            self.log_event(
                "run_started",
                run_id=self.run_id,
                run_name=self.run_name,
                run_dir=str(self.run_dir),
            )

    @staticmethod
    def _normalize_logs(logs: Optional[Any]):
        if logs is None:
            return ["train", "valid"], {}

        if isinstance(logs, Mapping):
            names = []
            headers = {}
            for name, header in logs.items():
                channel_name = SimpleLogger2._validate_log_name(str(name))
                if channel_name not in names:
                    names.append(channel_name)
                if header:
                    headers[channel_name] = [str(field) for field in header]
            return names, headers

        if isinstance(logs, str):
            return [SimpleLogger2._validate_log_name(logs)], {}

        if isinstance(logs, Iterable):
            names = []
            for name in logs:
                channel_name = SimpleLogger2._validate_log_name(str(name))
                if channel_name not in names:
                    names.append(channel_name)
            return names, {}

        raise TypeError("logs must be None, a string, a sequence, or a mapping.")

    @staticmethod
    def _validate_log_name(name: str) -> str:
        safe_name = RunLoggerBase._safe_name(name)
        if not safe_name:
            raise ValueError("Log name cannot be empty.")
        return safe_name

    def __getitem__(self, name: str) -> SimpleLogChannel:
        return self._channels[self._validate_log_name(name)]

    def __contains__(self, name: str) -> bool:
        return self._validate_log_name(name) in self._channels

    def keys(self):
        return self._channels.keys()

    def log_global_params(
        self,
        params: Mapping[str, Any],
        title: str = "GLOBAL PARAMETERS",
    ):
        if not self._is_main_process():
            return

        for name in self.log_names:
            self._write_info(name, title)
        for key, value in params.items():
            info_line = f"{key}: {self._format_value(value)}"
            for name in self.log_names:
                self._write_info(name, info_line)
        self.log_event("global_params_written")

    def log_system_info(
        self,
        title: str = "SYSTEM INFORMATION",
        include_git: bool = True,
        include_packages: bool = True,
        include_all_packages: bool = False,
        package_names: Optional[Sequence[str]] = None,
    ):
        """
        Record reproducibility-oriented system and environment information.

        This intentionally avoids dumping all environment variables because
        shells often contain credentials, tokens, and private paths.
        """
        info = self.collect_system_info(
            include_git=include_git,
            include_packages=include_packages,
            include_all_packages=include_all_packages,
            package_names=package_names,
        )
        self.log_info_block(title, info, prefix="E")
        self.log_event("system_info_written")

    def log_argparse_params(
        self,
        args: Any,
        title: str = "ARGPARSE PARAMETERS",
    ):
        """
        Record parsed argparse parameters.

        Pass an argparse.Namespace from parser.parse_args(). Mappings and
        simple objects with __dict__ are also accepted for convenience.
        """
        params = self._normalize_params(args)
        self.log_info_block(title, params, prefix="P")
        self.log_event("argparse_params_written")

    def log_params(
        self,
        params: Any,
        title: str = "PARAMETERS",
    ):
        """
        Record manually supplied run parameters with [P] lines.
        """
        self.log_info_block(title, self._normalize_params(params), prefix="P")
        self.log_event("params_written")

    def log_info_block(
        self,
        title: str,
        params: Mapping[str, Any],
        prefix: str = "I",
    ):
        if not self._is_main_process():
            return

        for name in self.log_names:
            self._write_prefixed_message(name, prefix, title)
        for key, value in params.items():
            info_line = f"{key}: {self._format_value(value)}"
            for name in self.log_names:
                self._write_prefixed_message(name, prefix, info_line)

    @staticmethod
    def _normalize_params(params: Any) -> Mapping[str, Any]:
        if isinstance(params, Mapping):
            return dict(params)
        if hasattr(params, "_actions"):
            normalized = {}
            for action in params._actions:
                dest = getattr(action, "dest", None)
                if not dest or dest == "help":
                    continue
                normalized[dest] = getattr(action, "default", None)
            return normalized
        if hasattr(params, "__dict__"):
            return vars(params)
        raise TypeError(
            "params must be a mapping, argparse.Namespace, "
            "argparse.ArgumentParser, or object with __dict__."
        )

    def collect_system_info(
        self,
        include_git: bool = True,
        include_packages: bool = True,
        include_all_packages: bool = False,
        package_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_base_prefix": sys.base_prefix,
            "python_is_virtualenv": sys.prefix != sys.base_prefix,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "user": self._safe_get_user(),
            "process_id": os.getpid(),
            "current_working_directory": os.getcwd(),
        }

        if include_git:
            info.update(self._collect_git_info())

        info.update(self._collect_torch_info())

        if include_packages:
            info.update(
                self._collect_package_info(
                    package_names=package_names,
                    include_all_packages=include_all_packages,
                )
            )

        return info

    @staticmethod
    def _safe_get_user() -> str:
        try:
            return getuser()
        except Exception:
            return ""

    @staticmethod
    def _default_package_names(package_names: Optional[Sequence[str]]) -> Sequence[str]:
        if package_names is not None:
            return package_names
        return (
            "torch",
            "torchvision",
            "torchaudio",
            "torchtext",
            "torchmetrics",
            "pytorch-lightning",
            "lightning",
            "numpy",
            "scipy",
            "pandas",
            "scikit-learn",
            "scikit-image",
            "opencv-python",
            "opencv-contrib-python",
            "albumentations",
            "matplotlib",
            "Pillow",
            "tensorboard",
            "wandb",
            "mlflow",
            "transformers",
            "datasets",
            "accelerate",
            "diffusers",
            "timm",
            "einops",
            "xformers",
            "deepspeed",
            "apex",
            "bitsandbytes",
            "onnx",
            "onnxruntime",
            "h5py",
            "zarr",
            "netCDF4",
            "segyio",
            "obspy",
            "tensorflow",
            "keras",
            "jax",
            "jaxlib",
            "flax",
            "optax",
        )

    @staticmethod
    def _get_package_version(package_name: str) -> Optional[str]:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return None

    def _collect_package_info(
        self,
        package_names: Optional[Sequence[str]],
        include_all_packages: bool,
    ) -> Dict[str, Any]:
        packages: Dict[str, str] = {}

        if include_all_packages:
            for distribution in metadata.distributions():
                name = distribution.metadata.get("Name")
                if not name:
                    continue
                packages[name] = distribution.version
            package_selection = "all_installed"
        else:
            for package_name in self._default_package_names(package_names):
                version = self._get_package_version(package_name)
                if version is not None:
                    packages[package_name] = version
            package_selection = (
                "custom" if package_names is not None else "deep_learning_defaults"
            )

        info: Dict[str, Any] = {
            "package_selection": package_selection,
        }

        if include_all_packages:
            info["installed_package_count"] = len(packages)
            for name in sorted(packages, key=str.lower):
                safe_name = self._safe_package_key(name)
                info[f"package_{safe_name}"] = packages[name]
        else:
            monitored_names = list(self._default_package_names(package_names))
            installed_count = 0
            missing_count = 0
            for name in monitored_names:
                safe_name = self._safe_package_key(name)
                version = packages.get(name)
                if version is None:
                    info[f"package_{safe_name}"] = "not_installed"
                    missing_count += 1
                else:
                    info[f"package_{safe_name}"] = version
                    installed_count += 1
            info["monitored_package_count"] = len(monitored_names)
            info["installed_package_count"] = installed_count
            info["missing_package_count"] = missing_count
        return info

    @staticmethod
    def _safe_package_key(package_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", package_name).strip("_")

    @staticmethod
    def _run_git_command(args: Sequence[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=os.getcwd(),
                capture_output=True,
                check=False,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def _collect_git_info(self) -> Dict[str, Any]:
        inside_work_tree = self._run_git_command(
            ["rev-parse", "--is-inside-work-tree"]
        )
        if inside_work_tree != "true":
            return {"git_available": False}

        status = self._run_git_command(["status", "--porcelain"])
        return {
            "git_available": True,
            "git_branch": self._run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"]
            ),
            "git_commit": self._run_git_command(["rev-parse", "HEAD"]),
            "git_dirty": bool(status),
        }

    @staticmethod
    def _collect_torch_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            import torch
        except ModuleNotFoundError:
            return {"torch_available": False}

        if not hasattr(torch, "__version__"):
            return {"torch_available": False}

        info["torch_available"] = True
        info["torch_version"] = getattr(torch, "__version__", "")
        cuda = getattr(torch, "cuda", None)
        version = getattr(torch, "version", None)
        backends = getattr(torch, "backends", None)

        cuda_available = bool(cuda and cuda.is_available())
        info["cuda_available"] = cuda_available
        info["cuda_version"] = getattr(version, "cuda", None)
        info["cudnn_version"] = None
        if backends is not None and getattr(backends, "cudnn", None) is not None:
            info["cudnn_version"] = backends.cudnn.version()

        info["gpu_count"] = cuda.device_count() if cuda else 0
        if info["cuda_available"]:
            gpu_names = [
                cuda.get_device_name(index)
                for index in range(cuda.device_count())
            ]
            info["gpu_names"] = ", ".join(gpu_names)
            info["current_cuda_device"] = cuda.current_device()
        return info

    def log_train(self, message: Optional[str] = None, **fields: Any):
        self._log_curve_row("train", message, fields)

    def log_valid(self, message: Optional[str] = None, **fields: Any):
        self._log_curve_row("valid", message, fields)

    def log_event(
        self,
        event: str,
        level: int = logging.INFO,
        **fields: Any,
    ):
        event_fields = {"event": event}
        event_fields.update(fields)
        self._write_prefixed(self.event_log, "I", event_fields, level=level)

    def log_train_header(self, fields: Sequence[str]):
        self._write_header("train", fields)

    def log_valid_header(self, fields: Sequence[str]):
        self._write_header("valid", fields)

    def log_train_epoch(
        self,
        epoch: int,
        metrics: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ):
        record = {"epoch": epoch}
        if metrics:
            record.update(metrics)
        record.update(fields)
        self.log_train(**record)

    def log_valid_epoch(
        self,
        epoch: int,
        metrics: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ):
        record = {"epoch": epoch}
        if metrics:
            record.update(metrics)
        record.update(fields)
        self.log_valid(**record)

    def _write_info(self, channel: str, message: str, level: int = logging.INFO):
        if not self._is_main_process():
            return
        self._write_prefixed_message(channel, "I", message, level=level)

    def _write_prefixed_message(
        self,
        channel: str,
        prefix: str,
        message: str,
        level: int = logging.INFO,
    ):
        if not self._is_main_process():
            return
        self._log(channel, level, f"[{prefix}] {message}")

    def _write_header(self, channel: str, fields: Sequence[str]):
        if not self._is_main_process():
            return
        header = [str(field) for field in fields]
        self._headers[channel] = header
        self._log(channel, logging.INFO, "[H] " + " ".join(["log_index", *header]))

    def _write_log_values(self, channel: str, values: Sequence[Any]):
        if not self._is_main_process():
            return
        log_index = self._log_counters.get(channel, 0)
        formatted_values = [
            f"{self._format_value(value):>{self.log_value_width}}"
            for value in [log_index, *values]
        ]
        self._log(channel, logging.INFO, "[L] " + " ".join(formatted_values))
        self._log_counters[channel] = log_index + 1

    def _write_prefixed(
        self,
        channel: str,
        prefix: str,
        fields: Mapping[str, Any],
        level: int = logging.INFO,
    ):
        if not self._is_main_process():
            return
        self._log(channel, level, f"[{prefix}] {self._format_record(None, fields)}")

    def _log_curve_row(
        self,
        channel: str,
        message: Optional[str],
        fields: Mapping[str, Any],
    ):
        if not self._is_main_process():
            return

        if message and not fields:
            self._write_info(channel, message)
            return

        record = dict(fields)
        if message:
            record = {"message": message, **record}

        header = self._configured_headers.get(channel, list(record.keys()))
        if self._headers.get(channel) != header:
            self._write_header(channel, header)

        self._write_log_values(
            channel,
            [record.get(key, "") for key in self._headers[channel]],
        )


if __name__ == "__main__":
    logger = SimpleLogger(
        log_dir="../logs",
        log_file="train.log",
        level=logging.INFO,
        overwrite=True,
        append=False,
    )
    logger.info("Start training")
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
