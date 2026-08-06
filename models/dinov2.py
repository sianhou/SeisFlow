import copy

import torch
from torch import nn

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
        encoder_dtype = next(self.encoder.parameters()).dtype
        x = x.to(dtype=encoder_dtype)
        feature = self.encoder.forward_features(x)["x_norm_patchtokens"]
        return feature.to(torch.bfloat16)
