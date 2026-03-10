import argparse
import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from core.dataset import SegyDataset, PatchDataset
from core.patching import extract_overlapping_patches_2d
from core.transforms import SliceLastDimension, ClipFirstChannel, ScaleFirstChannel


def create_parser():
    parser = argparse.ArgumentParser(description='Create patch dataset')
    parser.add_argument("--dataset", default="/disk03/hsa/SeisFlow/ma2+GathAP.sgy",
                        help="Segy data for training / testing")
    parser.add_argument("--patch_size", default=32, type=int, )
    parser.add_argument("--overlap_size", default=16, type=int, )
    parser.add_argument("--output_dir", default="./dataset_train", )
    parser.add_argument("--normalize", action="store_true", )
    return parser


def create_data(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # dataset
    transform = transforms.Compose([
        SliceLastDimension(0, 1501),
        ClipFirstChannel(-2, 2),
        ScaleFirstChannel(0.5),
        transforms.Resize((256, 256)),
    ])
    dataset = SegyDataset("ma2+GathAP.sgy", transform=transform)

    patch_size = (args.patch_size, args.patch_size)
    overlap_size = (args.overlap_size, args.overlap_size)

    for i in range(len(dataset)):
        patches, positions, original_shape = extract_overlapping_patches_2d(dataset[i][0].numpy(),
                                                                             patch_size=patch_size,
                                                                             overlap=overlap_size)
        # Save to NPZ file
        output_file = os.path.join(args.output_dir, f"patches_{i:04d}.npz")

        # Convert positions list to numpy array for saving
        positions_array = np.array(positions)

        np.savez_compressed(
            output_file,
            patches=patches,
            positions=positions_array,
            original_shape=original_shape,
            patch_size=patch_size,
            overlap_size=overlap_size
        )

        print(f"Saved {output_file} with {len(patches)} patches")


def test_data(args):
    print("Creating patch dataset")
    pd = PatchDataset(data_path=args.output_dir)

    loader = DataLoader(pd, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    x = next(iter(loader))
    print(x.shape)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    create_data(args)
    test_data(args)
