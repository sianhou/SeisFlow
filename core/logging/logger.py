import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import re
import time
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
        output_dir: str = "./results",
        log_id: Optional[str] = None,
        level: int = logging.INFO,
        overwrite: bool = False,
        append: bool = False,
        console: bool = True,
    ):
        if overwrite and append:
            raise ValueError("overwrite and append cannot both be True.")

        self.output_dir = Path(output_dir)
        self.log_id = log_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.level = level
        self.overwrite = overwrite
        self.append = append
        self.console = console
        self._loggers: Dict[str, logging.Logger] = {}
        self.run_dir = self.output_dir / self.log_id

        if self._should_log():
            self._prepare_run_dir()

    @staticmethod
    def _safe_name(value: str) -> str:
        value = value.strip()
        value = re.sub(r'[<>:"/\\|?*\s]+', "_", value)
        value = value.strip("._")
        return value or "run"

    def _prepare_run_dir(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _file_mode(self) -> str:
        if self.append:
            return "a"
        return "w"

    def _should_log(self) -> bool:
        """
        Return whether this logger instance should emit records.

        The base run logger is process-agnostic and always writes. Distributed
        variants can override this hook to suppress specific local ranks.
        """
        return True

    def _build_logger(self, channel: str, file_name: str) -> logging.Logger:
        if channel in self._loggers:
            return self._loggers[channel]

        logger_name = (
            f"{self.__class__.__name__}.{self.log_id}.{channel}.{id(self)}"
        )
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        logger.propagate = False

        formatter = logging.Formatter(fmt="%(message)s")

        file_path = self.run_dir / file_name
        if file_path.exists() and not self.overwrite and not self.append:
            raise FileExistsError(
                f"Log file '{file_path}' already exists. "
                "Set overwrite=True to overwrite or append=True to append."
            )
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
        if not self._should_log():
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


class SimpleLogger2(RunLoggerBase):
    """
    Run-scoped training logger with one LAS-like log file.

    By default, files created in each run directory:
        - log.txt

    Example:
        logger = SimpleLogger2(output_dir="./logs", log_id="unet")
        logger.log_info_block("GLOBAL PARAMETERS", {"epochs": 100, "batch_size": 16})
        logger.log_train(epoch=1, loss=0.23, lr=1e-4)
        logger.log_valid(epoch=1, loss=0.31, psnr=24.7)
        logger.log_event("checkpoint_saved", path="model_epoch_00001.pth")
    """

    def __init__(
        self,
        output_dir: str = "./results",
        log_id: Optional[str] = None,
        level: int = logging.INFO,
        overwrite: bool = False,
        append: bool = False,
        console: bool = True,
        logs: Optional[Any] = None,
        log_file: str = "log.txt",
        log_value_width: int = 10,
    ):
        if log_value_width < 1:
            raise ValueError("log_value_width must be >= 1.")

        super().__init__(
            output_dir=output_dir,
            log_id=log_id,
            level=level,
            overwrite=overwrite,
            append=append,
            console=console,
        )
        self._header: List[str] = []
        self._log_counter = 0
        self.log_name = "log"
        self.log_file = log_file
        self.log_value_width = log_value_width
        self._configured_header = self._normalize_logs(logs)

        if self._should_log():
            self._build_logger(self.log_name, self.log_file)
            self.log_event(
                "run_started",
                log_id=self.log_id,
                run_dir=str(self.run_dir),
            )

    @staticmethod
    def _normalize_logs(logs: Optional[Any]):
        if logs is None:
            return []

        if isinstance(logs, str):
            return [logs]

        if isinstance(logs, Iterable):
            return [str(field) for field in logs]

        raise TypeError("logs must be None, a string, or one sequence of fields.")

    def log(self, prefix: str = "L", **fields: Any):
        self._log_curve_row(prefix, fields)

    def log_train(self, prefix: str = "T", **fields: Any):
        self.log(prefix=prefix, **fields)

    def log_valid(self, prefix: str = "V", **fields: Any):
        self.log(prefix=prefix, **fields)

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
        self.log_info_block(title, info)

    def log_info_block(
        self,
        title: str,
        params: Any,
        blank_lines: int = 2,
    ):
        self._write_info(title)
        for key, value in self._normalize_params(params).items():
            self._write_info(f"{key}: {self._format_value(value)}")
        for _ in range(blank_lines):
            self._write_info_blank_line()

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

        info.update(self._collect_hardware_info())

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
    def _bytes_to_gib(value: int) -> float:
        return value / 1024**3

    @staticmethod
    def _run_system_command(args: Sequence[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                list(args),
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

    def _collect_hardware_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "cpu_model": self._collect_cpu_model(),
            "cpu_physical_cores": self._collect_physical_cpu_cores(),
            "cpu_logical_threads": os.cpu_count(),
        }
        info.update(self._collect_memory_info())
        info.update(self._collect_disk_info())
        info.update(self._collect_apple_info())
        return info

    def _collect_cpu_model(self) -> str:
        system = platform.system()
        if system == "Darwin":
            return self._run_system_command(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ) or platform.processor()
        if system == "Linux":
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                    for line in cpuinfo:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        if system == "Windows":
            return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
        return platform.processor()

    def _collect_physical_cpu_cores(self) -> Optional[int]:
        try:
            import psutil

            return psutil.cpu_count(logical=False)
        except (ImportError, AttributeError):
            pass

        system = platform.system()
        if system == "Darwin":
            value = self._run_system_command(["sysctl", "-n", "hw.physicalcpu"])
            return int(value) if value and value.isdigit() else None
        if system == "Linux":
            core_ids = set()
            physical_id = "0"
            core_id = None
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                    for line in [*cpuinfo, "\n"]:
                        if line.startswith("physical id"):
                            physical_id = line.split(":", 1)[1].strip()
                        elif line.startswith("core id"):
                            core_id = line.split(":", 1)[1].strip()
                        elif not line.strip() and core_id is not None:
                            core_ids.add((physical_id, core_id))
                            core_id = None
            except OSError:
                return None
            return len(core_ids) or None
        return None

    def _collect_memory_info(self) -> Dict[str, Any]:
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "memory_total_gib": self._bytes_to_gib(memory.total),
                "memory_used_gib": self._bytes_to_gib(memory.used),
                "memory_available_gib": self._bytes_to_gib(memory.available),
                "memory_percent_used": memory.percent,
            }
        except (ImportError, AttributeError):
            pass

        total = None
        available = None
        if platform.system() == "Linux":
            values: Dict[str, int] = {}
            try:
                with open("/proc/meminfo", encoding="utf-8") as meminfo:
                    for line in meminfo:
                        key, value = line.split(":", 1)
                        values[key] = int(value.strip().split()[0]) * 1024
            except (OSError, ValueError):
                values = {}
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
        elif platform.system() == "Darwin":
            value = self._run_system_command(["sysctl", "-n", "hw.memsize"])
            total = int(value) if value and value.isdigit() else None
            vm_stat = self._run_system_command(["vm_stat"])
            if vm_stat:
                available = self._parse_macos_available_memory(vm_stat)

        if total is None:
            return {}
        used = total - available if available is not None else None
        info: Dict[str, Any] = {"memory_total_gib": self._bytes_to_gib(total)}
        if used is not None:
            info["memory_used_gib"] = self._bytes_to_gib(used)
            info["memory_available_gib"] = self._bytes_to_gib(available)
            info["memory_percent_used"] = used / total * 100
        return info

    @staticmethod
    def _parse_macos_available_memory(vm_stat: str) -> Optional[int]:
        lines = vm_stat.splitlines()
        if not lines:
            return None

        page_size_match = re.search(r"page size of (\d+) bytes", lines[0])
        if page_size_match is None:
            return None
        page_size = int(page_size_match.group(1))

        available_page_names = {
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        }
        available_pages = 0
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            if name not in available_page_names:
                continue
            page_count = value.strip().rstrip(".")
            if page_count.isdigit():
                available_pages += int(page_count)
        return available_pages * page_size

    def _collect_disk_info(self) -> Dict[str, Any]:
        try:
            disk = shutil.disk_usage(os.getcwd())
        except OSError:
            return {}
        return {
            "disk_path": os.getcwd(),
            "disk_total_gib": self._bytes_to_gib(disk.total),
            "disk_used_gib": self._bytes_to_gib(disk.used),
            "disk_free_gib": self._bytes_to_gib(disk.free),
            "disk_percent_used": disk.used / disk.total * 100 if disk.total else 0.0,
        }

    def _collect_apple_info(self) -> Dict[str, Any]:
        if platform.system() != "Darwin":
            return {}

        machine = platform.machine()
        info: Dict[str, Any] = {
            "apple_silicon": machine in {"arm64", "aarch64"},
            "apple_model": self._run_system_command(
                ["sysctl", "-n", "hw.model"]
            ),
            "apple_chip": self._run_system_command(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ),
        }
        memory_total = self._run_system_command(["sysctl", "-n", "hw.memsize"])
        if memory_total and memory_total.isdigit():
            info["apple_unified_memory_gib"] = self._bytes_to_gib(int(memory_total))
        performance_cores = self._run_system_command(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
        )
        efficiency_cores = self._run_system_command(
            ["sysctl", "-n", "hw.perflevel1.physicalcpu"]
        )
        if performance_cores and performance_cores.isdigit():
            info["apple_performance_cores"] = int(performance_cores)
        if efficiency_cores and efficiency_cores.isdigit():
            info["apple_efficiency_cores"] = int(efficiency_cores)
        return info

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
            gpu_names = []
            gpu_memory_total_gib = []
            for index in range(cuda.device_count()):
                name = cuda.get_device_name(index)
                total_memory = cuda.get_device_properties(index).total_memory
                total_memory_gib = total_memory / 1024**3
                gpu_names.append(name)
                gpu_memory_total_gib.append(f"{total_memory_gib:.6g}")
                info[f"gpu_{index}_name"] = name
                info[f"gpu_{index}_memory_total_gib"] = total_memory_gib
            info["gpu_names"] = ", ".join(gpu_names)
            info["gpu_memory_total_gib"] = ", ".join(gpu_memory_total_gib)
            info["current_cuda_device"] = cuda.current_device()

        mps = getattr(backends, "mps", None) if backends is not None else None
        info["mps_built"] = bool(mps and mps.is_built())
        info["mps_available"] = bool(mps and mps.is_available())
        return info

    def log_event(
        self,
        event: str,
        prefix: str = "I",
        level: int = logging.INFO,
        status: Optional[str] = None,
        **fields: Any,
    ):
        event_fields = {"event": event}
        event_fields.update(fields)
        status_name = status or logging.getLevelName(level).lower()
        self._write_prefixed(prefix, event_fields, level=level, status=status_name)

    def _write_info(self, message: str, level: int = logging.INFO):
        self._write_prefixed_message("I", message, level=level)

    def _write_info_blank_line(self, level: int = logging.INFO):
        self._log(self.log_name, level, "[I]")

    def _write_prefixed_message(
        self,
        prefix: str,
        message: str,
        level: int = logging.INFO,
    ):
        self._log(self.log_name, level, f"[{prefix}] {message}")

    def _write_header(self, fields: Sequence[str]):
        header = self._with_timestamp_column(fields)
        self._header = header
        self._log(self.log_name, logging.INFO, "[H] " + " ".join(["log_index", *header]))

    def _write_log_values(self, prefix: str, values: Sequence[Any]):
        line_prefix = self._validate_line_prefix(prefix)
        log_index = self._log_counter
        formatted_values = [
            f"{self._format_value(value):>{self.log_value_width}}"
            for value in [log_index, *values]
        ]
        self._log(self.log_name, logging.INFO, f"[{line_prefix}] " + " ".join(formatted_values))
        self._log_counter = log_index + 1

    def _write_prefixed(
        self,
        prefix: str,
        fields: Mapping[str, Any],
        level: int = logging.INFO,
        status: Optional[str] = None,
    ):
        status_text = f"{status} | " if status else ""
        self._log(
            self.log_name,
            level,
            f"[{prefix}] {status_text}{self._format_record(None, fields)}",
        )

    def _log_curve_row(
        self,
        prefix: str,
        fields: Mapping[str, Any],
    ):
        record = dict(fields)

        header = self._with_timestamp_column(self._configured_header or list(record.keys()))
        if self._header != header:
            self._write_header(header)

        record["timestamp"] = time.time_ns()
        self._write_log_values(
            prefix,
            [record.get(key, "") for key in self._header],
        )

    @staticmethod
    def _with_timestamp_column(fields: Sequence[str]) -> List[str]:
        header = [str(field) for field in fields]
        return header if "timestamp" in header else [*header, "timestamp"]

    @staticmethod
    def _validate_line_prefix(prefix: str) -> str:
        text = str(prefix)
        if len(text) != 1:
            raise ValueError("log prefix must be exactly one character.")
        if text in {"H", "I"}:
            raise ValueError("log prefix cannot be 'H' or 'I'.")
        return text


def resolve_distributed_log_id(log_id=None, distributed=False):
    """
    Resolve one shared log_id for all ranks in a distributed job.

    Rank 0 creates the timestamp-based id, then broadcasts it to every rank.
    """
    if log_id is not None:
        return log_id
    if not distributed:
        return None

    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_id_holder = [log_id]
    dist.broadcast_object_list(log_id_holder, src=0)
    return log_id_holder[0]


def collect_node_info(
        rank=0,
        world_size=1,
        local_rank=0,
        hostname=None,
        environ=None,
):
    environ = os.environ if environ is None else environ
    return {
        "hostname": hostname or socket.gethostname(),
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_world_size": environ.get("LOCAL_WORLD_SIZE", ""),
        "node_rank": environ.get("GROUP_RANK", ""),
        "master_addr": environ.get("MASTER_ADDR", ""),
        "master_port": environ.get("MASTER_PORT", ""),
    }


class DistributedSimpleLogger2(SimpleLogger2):
    """
    Node-scoped DDP logger with a SimpleLogger2-like interface.

    All ranks should instantiate this class. It resolves a shared run ``log_id``
    across the distributed job, writes one log file per hostname, and only the
    local rank 0 process on each node emits records.
    """

    def __init__(
            self,
            output_dir: str = "./results",
            log_id: Optional[str] = None,
            distributed: bool = False,
            rank: int = 0,
            world_size: int = 1,
            local_rank: int = 0,
            hostname: Optional[str] = None,
            environ: Optional[Mapping[str, str]] = None,
            log_file: Optional[str] = None,
            **kwargs,
    ):
        self.node_info = collect_node_info(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            hostname=hostname,
            environ=environ,
        )
        resolved_log_id = resolve_distributed_log_id(
            log_id=log_id,
            distributed=distributed,
        )
        if log_file is None:
            if distributed:
                log_file = f"log_{self.node_info['hostname']}.txt"
            else:
                log_file = "log.txt"

        super().__init__(
            output_dir=output_dir,
            log_id=resolved_log_id,
            log_file=log_file,
            **kwargs,
        )
        self.distributed = distributed
        self.log_file = log_file

    def _is_global_main_process(self) -> bool:
        return int(self.node_info["rank"]) == 0

    def _is_local_main_process(self) -> bool:
        return int(self.node_info["local_rank"]) == 0

    def _should_log(self):
        return self._is_local_main_process()

    def log_node_info(self, title: str = "NODE INFORMATION"):
        self.log_info_block(title, self.node_info)


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
