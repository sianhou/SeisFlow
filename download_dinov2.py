"""Download a DINOv2 checkpoint into a local Torch Hub directory."""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DINOv2(nn.Module):
    """DINOv2 teacher wrapper with an explicit local Torch Hub directory."""

    def __init__(
            self,
            model_name="dinov2_vitb14",
            base_patch_size=16,
            hub_dir=None,
            force_reload=False,
    ):
        super().__init__()
        self.hub_dir = hub_dir
        torch.hub.set_dir(str(self.hub_dir))

        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            trust_repo=True,
            force_reload=force_reload,
        )
        self.encoder = self.encoder.to(torch.bfloat16)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = {}
        self.base_patch_size = base_patch_size
        self.encoder.eval()

    @staticmethod
    def _normalize(x):
        mean = torch.tensor(
            IMAGENET_DEFAULT_MEAN,
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            IMAGENET_DEFAULT_STD,
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def forward(self, x, resize=True):
        _, _, height, width = x.shape
        x = self._normalize(x)
        if resize:
            x = torch.nn.functional.interpolate(
                x,
                (
                    int(14 * height / self.base_patch_size),
                    int(14 * width / self.base_patch_size),
                ),
                mode="bicubic",
            )
        feature = self.encoder.forward_features(x)["x_norm_patchtokens"]
        return feature.to(torch.bfloat16)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download a DINOv2 model into the current directory."
    )
    parser.add_argument(
        "--model-name",
        default="dinov2_vitb14",
        choices=(
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ),
        help="DINOv2 model exposed by facebookresearch/dinov2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Torch Hub root directory; defaults to the current directory.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Redownload the DINOv2 repository instead of using the cache.",
    )
    return parser


def download_dinov2(model_name="dinov2_vitb14", output_dir=None, force_reload=False):
    """Download and instantiate DINOv2, returning the local cache paths."""
    hub_dir = Path.cwd() if output_dir is None else Path(output_dir)
    hub_dir = hub_dir.expanduser().resolve()
    print(f"Torch Hub directory: {hub_dir}")
    print(f"Model: {model_name}")
    print("Downloading DINOv2 repository and pretrained weights...")
    model = DINOv2(
        model_name=model_name,
        hub_dir=hub_dir,
        force_reload=force_reload,
    )

    print("DINOv2 download completed.")
    print(f"Model embedding dimension: {getattr(model.encoder, 'embed_dim', 'unknown')}")
    print(f"Patch size: {getattr(model.encoder.patch_embed, 'patch_size', 'unknown')}")
    print(f"Repository cache: {model.hub_dir / 'hub'}")
    print(f"Checkpoint cache: {model.hub_dir / 'hub' / 'checkpoints'}")
    return model


def main():
    args = build_parser().parse_args()
    download_dinov2(
        model_name=args.model_name,
        output_dir=args.output_dir,
        force_reload=args.force_reload,
    )


if __name__ == "__main__":
    main()
