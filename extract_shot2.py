"""Extract every SEG-Y shot into an independent seismic NPY file."""

import argparse
import os

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract all SEG-Y shots as independent seismic NPY files."
    )
    parser.add_argument("--segy", required=True, help="Input SEG-Y file.")
    parser.add_argument(
        "--output_dir",
        default="./extracted_shots",
        help="Directory for independent shot NPY files.",
    )
    parser.add_argument(
        "--clip",
        nargs=2,
        default=None,
        type=float,
        metavar=("VMIN", "VMAX"),
        help="Clip seismic amplitudes to [VMIN, VMAX].",
    )
    parser.add_argument(
        "--slice",
        nargs=2,
        default=[0, 0],
        type=int,
        metavar=("START", "END"),
        help="Slice the sample axis as [START, END). Use 0 0 to disable.",
    )
    return parser


def validate_args(args):
    if args.clip is not None and args.clip[0] > args.clip[1]:
        raise ValueError("--clip VMIN must be less than or equal to VMAX.")
    if args.slice[0] < 0 or args.slice[1] < 0:
        raise ValueError("--slice values must be non-negative.")
    if args.slice != [0, 0] and args.slice[0] >= args.slice[1]:
        raise ValueError("--slice must be 0 0 or satisfy START < END.")


def extract_shot(segy_dataset, shot_index, clip_range, slice_range):
    sample = segy_dataset[shot_index]
    shot = sample[0].numpy().astype(np.float32, copy=True)

    # Keep the same preprocessing order as build_shot_dataset2.py.
    if clip_range is not None:
        shot = np.clip(shot, clip_range[0], clip_range[1])
    if slice_range != [0, 0]:
        start, end = slice_range
        shot = shot[:, start:end]

    return shot.astype(np.float32, copy=False)


def main():
    args = build_parser().parse_args()
    validate_args(args)

    from core.dataset import SegyDataset

    dataset = SegyDataset(args.segy)
    os.makedirs(args.output_dir, exist_ok=True)

    for shot_index in range(len(dataset)):
        shot = extract_shot(
            dataset,
            shot_index,
            clip_range=args.clip,
            slice_range=args.slice,
        )
        output_file = os.path.join(args.output_dir, f"shot_{shot_index:04d}.npy")
        np.save(output_file, shot)
        print(
            f"Saved {output_file} shape={shot.shape}, "
            f"min={float(shot.min()):.7g}, max={float(shot.max()):.7g}"
        )


if __name__ == "__main__":
    main()
