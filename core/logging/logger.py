import logging
import os
import sys

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


if __name__ == "__main__":
    logger = SimpleLogger(log_dir="../logs", log_file="train.log", level=logging.INFO, overwrite=True, append=False)
    logger.info("Start training")
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
