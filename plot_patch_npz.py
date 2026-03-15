import argparse
from pathlib import Path

import numpy as np

from core.visualization import plot_seismic_grid


def create_parser():
    parser = argparse.ArgumentParser(
        description="Plot consecutive patch data from an NPZ file into a square PNG grid."
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Input NPZ file path.",
    )
    parser.add_argument(
        "--data_name",
        default="patches",
        help="Array name inside the NPZ file. If missing, available arrays will be printed.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for consecutive plotting.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2,
        help="Grid size. Example: 3 means 3x3.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. If omitted, a default PNG name will be generated next to the NPZ file.",
    )
    parser.add_argument(
        "--cmap",
        default="seismic",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--origin",
        default="upper",
        choices=["upper", "lower"],
        help="Image origin used by matplotlib.",
    )
    parser.add_argument(
        "--no_transpose",
        action="store_true",
        help="Disable transpose before plotting.",
    )
    return parser


def format_npz_contents(npz_file):
    lines = []
    for key in npz_file.files:
        value = npz_file[key]
        lines.append(
            f"- {key}: shape={getattr(value, 'shape', None)}, dtype={getattr(value, 'dtype', None)}"
        )
    return "\n".join(lines)


def prepare_plot_array(data, data_name):
    array = np.asarray(data)

    if array.ndim == 2:
        return array[np.newaxis, :, :]

    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0, :, :]

    if array.ndim != 3:
        raise ValueError(
            f"Expected `{data_name}` to have shape [N,H,W], [N,1,H,W], or [H,W], got {array.shape}"
        )

    return array


def main():
    args = create_parser().parse_args()
    npz_path = Path(args.npz)

    with np.load(npz_path) as npz_file:
        if args.data_name not in npz_file.files:
            print(f"Data name `{args.data_name}` not found in {npz_path}.")
            print("Available arrays:")
            print(format_npz_contents(npz_file))
            return

        imgs = prepare_plot_array(npz_file[args.data_name], args.data_name)

    if args.start_index < 0:
        raise ValueError(f"`start_index` must be >= 0, got {args.start_index}")

    if args.start_index >= imgs.shape[0]:
        raise IndexError(
            f"`start_index` {args.start_index} is out of range for `{args.data_name}` with {imgs.shape[0]} items"
        )

    imgs = imgs[args.start_index:args.start_index + args.size * args.size]

    output = args.output
    if output is None:
        output = npz_path.with_name(
            f"{npz_path.stem}_{args.data_name}_{args.start_index}_{args.size}x{args.size}.png"
        )

    plot_seismic_grid(
        imgs=imgs,
        fig_name=output,
        title=f"{npz_path.name} | {args.data_name} | start={args.start_index}",
        size=args.size,
        cmap=args.cmap,
        origin=args.origin,
        transpose=not args.no_transpose,
    )


if __name__ == "__main__":
    main()
