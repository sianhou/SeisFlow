import argparse
import os

import torch

from core.logger import BaseLogger
from training import distributed_mode


def create_parser():
    parser = argparse.ArgumentParser(description='Synthetic seismic dataset training')
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * gpus",
    )
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Path where to save, empty for no saving",
    )
    return parser


def train(args):
    logger = BaseLogger(log_dir=args.output_dir, overwrite=True)
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))

    distributed_mode.init_distributed_mode(args)

    device = torch.device(args.device)
    pass


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    train(args)
