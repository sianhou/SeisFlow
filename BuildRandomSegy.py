#!/usr/bin/env python3
"""Create a SEG-Y file with Gaussian random trace samples.

The output keeps the input SEG-Y structure and all headers unchanged. Only
trace sample values are replaced. The default distribution is N(0, 1),
matching ``torch.randn_like`` used by ``DiTSeisDimRecon.py``.

Example:
    python BuildRandomSegy.py \
        --segy input.sgy \
        --output random.sgy \
        --seed 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a same-size SEG-Y file with independent Gaussian random "
            "trace samples. SEG-Y headers are copied unchanged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--segy", required=True, help="Input SEG-Y file.")
    parser.add_argument("--output", required=True, help="Output random SEG-Y file.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible random data.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.0,
        help="Mean of the Gaussian distribution.",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1.0,
        help="Standard deviation of the Gaussian distribution.",
    )
    parser.add_argument(
        "--chunk-traces",
        type=int,
        default=256,
        help="Number of traces generated and written in one chunk.",
    )
    return parser


def validate_args(args):
    input_path = Path(args.segy)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input SEG-Y file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must be different from --segy.")
    if args.std <= 0:
        raise ValueError("--std must be positive.")
    if args.chunk_traces <= 0:
        raise ValueError("--chunk-traces must be positive.")


def create_random_segy(input_path, output_path, seed, mean, std, chunk_traces):
    import segyio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)

    rng = np.random.default_rng(seed)
    with segyio.open(str(input_path), "r", ignore_geometry=True) as source:
        trace_count = int(source.tracecount)
        sample_count = len(source.samples)
        sample_interval = getattr(source, "sample_interval", None)

    with segyio.open(str(output_path), "r+", ignore_geometry=True) as target:
        for start in range(0, trace_count, chunk_traces):
            end = min(start + chunk_traces, trace_count)
            random_block = rng.normal(
                loc=mean,
                scale=std,
                size=(end - start, sample_count),
            ).astype(np.float32, copy=False)
            for trace_index, trace in enumerate(random_block, start=start):
                target.trace[trace_index] = trace
            target.flush()

    return trace_count, sample_count, sample_interval


def main():
    args = build_parser().parse_args()
    validate_args(args)
    trace_count, sample_count, sample_interval = create_random_segy(
        input_path=args.segy,
        output_path=args.output,
        seed=args.seed,
        mean=args.mean,
        std=args.std,
        chunk_traces=args.chunk_traces,
    )

    print(f"Saved random SEG-Y: {args.output}")
    print(f"Shape: traces={trace_count}, samples={sample_count}")
    if sample_interval is not None:
        print(f"Sample interval: {sample_interval}")
    print(f"Distribution: Normal(mean={args.mean}, std={args.std}), seed={args.seed}")


if __name__ == "__main__":
    main()
