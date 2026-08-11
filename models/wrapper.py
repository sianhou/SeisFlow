from pathlib import Path

import math
import torch
from diffusers.models import AutoencoderKL, DiTTransformer2DModel
from torch import nn

from core.training.model_utils import count_model_parameters
from flow_matching.utils import ModelWrapper
from models.pixeldit import PixDiT
from diffusers.training_utils import EMAModel as EMA

TRAINING_STATE_NAME = "training_state.pth"
EMA_DIR_NAME = "ema"

DIT_TRANSFORMER_2D_CONFIGS = {
    "DiT_XL_2": {
        "num_layers": 28,
        "attention_head_dim": 72,
        "num_attention_heads": 16,
        "patch_size": 2,
    },
    "DiT_XL_4": {
        "num_layers": 28,
        "attention_head_dim": 72,
        "num_attention_heads": 16,
        "patch_size": 4,
    },
    "DiT_XL_8": {
        "num_layers": 28,
        "attention_head_dim": 72,
        "num_attention_heads": 16,
        "patch_size": 8,
    },
    "DiT_L_2": {
        "num_layers": 24,
        "attention_head_dim": 64,
        "num_attention_heads": 16,
        "patch_size": 2,
    },
    "DiT_L_4": {
        "num_layers": 24,
        "attention_head_dim": 64,
        "num_attention_heads": 16,
        "patch_size": 4,
    },
    "DiT_L_8": {
        "num_layers": 24,
        "attention_head_dim": 64,
        "num_attention_heads": 16,
        "patch_size": 8,
    },
    "DiT_B_2": {
        "num_layers": 12,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
        "patch_size": 2,
    },
    "DiT_B_4": {
        "num_layers": 12,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
        "patch_size": 4,
    },
    "DiT_B_8": {
        "num_layers": 12,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
        "patch_size": 8,
    },
    "DiT_S_2": {
        "num_layers": 12,
        "attention_head_dim": 64,
        "num_attention_heads": 6,
        "patch_size": 2,
    },
    "DiT_S_4": {
        "num_layers": 12,
        "attention_head_dim": 64,
        "num_attention_heads": 6,
        "patch_size": 4,
    },
    "DiT_S_8": {
        "num_layers": 12,
        "attention_head_dim": 64,
        "num_attention_heads": 6,
        "patch_size": 8,
    },
    "DiT_T_2": {
        "num_layers": 8,
        "attention_head_dim": 64,
        "num_attention_heads": 6,
        "patch_size": 2,
    },
    "DiT_T_4": {
        "num_layers": 8,
        "attention_head_dim": 64,
        "num_attention_heads": 6,
        "patch_size": 4,
    },
    "DiT_T_8": {
        "num_layers": 8,
        "attention_head_dim": 64,
        "num_attention_heads": 6,
        "patch_size": 8,
    },
}

Pixel_DiT_2D_CONFIGS = {
    "XL": {
        "num_groups": 16,
        "hidden_size": 1152,
        "patch_depth": 26,
        "pixel_depth": 4,
    },
    "L": {
        "num_groups": 16,
        "hidden_size": 1024,
        "patch_depth": 22,
        "pixel_depth": 4,
    },
    "S": {
        "num_groups": 12,
        "hidden_size": 768,
        "patch_depth": 12,
        "pixel_depth": 2,
    },
    "T": {
        "num_groups": 6,
        "hidden_size": 384,
        "patch_depth": 8,
        "pixel_depth": 2,
    },
}


class VelocityModel(ModelWrapper):
    """Adapt a conditional velocity model to the ODE solver interface."""

    def __init__(self, model):
        super().__init__(model)

    def forward(self, x, t, cfg_scale, label, concat_conditioning):
        del cfg_scale, label

        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])

        with torch.inference_mode():
            with torch.amp.autocast(
                    device_type=x.device.type,
                    enabled=x.device.type == "cuda",
            ):
                result = self.model(x, t, extra=concat_conditioning)
        return result.to(dtype=torch.float32)


class AutoencoderKLWrapper(nn.Module):
    def __init__(self, model: AutoencoderKL):
        super().__init__()
        self.model = model

    def forward(self, sample, sample_posterior=False, return_dict=True, generator=None):
        posterior = self.model.encode(sample).latent_dist
        if sample_posterior:
            latents = posterior.sample(generator=generator)
        else:
            latents = posterior.mode()
        reconstruction = self.model.decode(latents).sample

        outputs = {
            "recon": reconstruction,
            "mean": posterior.mean,
            "logvar": posterior.logvar,
        }
        if return_dict:
            return outputs
        return tuple(outputs.values())

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)

    def save_checkpoint(
            self,
            checkpoint_path,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            args=None,
            epoch=None,
    ):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = {
            "model": self.model.state_dict(),
            "model_config": dict(self.model.config),
        }
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None:
            checkpoint["amp_scaler"] = scaler.state_dict()
        if args is not None:
            checkpoint["args"] = vars(args)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
            self,
            checkpoint_path,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            device=None,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        map_location = device if device is not None else "cpu"

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        self.model.load_state_dict(checkpoint["model"])
        if optimizer is not None and checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and checkpoint.get("lr_scheduler"):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if scaler is not None and checkpoint.get("amp_scaler"):
            scaler.load_state_dict(checkpoint["amp_scaler"])

        return int(checkpoint.get("epoch", 0)), checkpoint

    @classmethod
    def from_pretrained(cls, save_directory, device=None, **kwargs):
        model = AutoencoderKL.from_pretrained(
            save_directory,
            local_files_only=True,
            **kwargs,
        )
        if device is not None:
            model = model.to(device)
        return cls(model)

    def save_training(
            self,
            save_directory,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            args=None,
            epoch=None,
    ):
        save_directory = Path(save_directory)
        self.save_pretrained(save_directory)

        training_state = {}
        if epoch is not None:
            training_state["epoch"] = epoch
        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            training_state["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None:
            training_state["amp_scaler"] = scaler.state_dict()
        if args is not None:
            training_state["args"] = vars(args)
        torch.save(training_state, save_directory / TRAINING_STATE_NAME)

    @classmethod
    def from_training(
            cls,
            save_directory,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            device=None,
    ):
        save_directory = Path(save_directory)
        training_state_path = save_directory / TRAINING_STATE_NAME
        if not training_state_path.is_file():
            raise FileNotFoundError(
                f"Training state file not found: {training_state_path}"
            )
        map_location = device if device is not None else "cpu"
        wrapper = cls.from_pretrained(save_directory, device=device)

        try:
            training_state = torch.load(
                training_state_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            training_state = torch.load(
                training_state_path,
                map_location=map_location,
            )

        if optimizer is not None and training_state.get("optimizer"):
            optimizer.load_state_dict(training_state["optimizer"])

        if lr_scheduler is not None and training_state.get("lr_scheduler"):
            lr_scheduler.load_state_dict(training_state["lr_scheduler"])

        if scaler is not None and training_state.get("amp_scaler"):
            scaler.load_state_dict(training_state["amp_scaler"])

        return wrapper, int(training_state.get("epoch", 0)), training_state


def build_autoencoder_kl_wrapper(
        input_size,
        latent_size,
        input_channels=1,
        output_channels=1,
        latent_channels=4,
        hidden_channels=32,
        channel_multipliers=None,
        device=None,
):
    compression = input_size // latent_size
    if (
            input_size % latent_size != 0
            or compression < 1
            or compression & (compression - 1) != 0
    ):
        raise ValueError("input_size / latent_size must be a positive power of two.")

    num_blocks = int(math.log2(compression)) + 1
    if channel_multipliers:
        channel_multipliers = tuple(channel_multipliers)
        if len(channel_multipliers) != num_blocks:
            raise ValueError(
                "channel_multipliers length must equal "
                "log2(input_size / latent_size) + 1."
            )
    else:
        channel_multipliers = tuple(min(2 ** idx, 8) for idx in range(num_blocks))

    block_out_channels = tuple(
        hidden_channels * multiplier for multiplier in channel_multipliers
    )
    autoencoder = AutoencoderKL(
        in_channels=input_channels,
        out_channels=output_channels,
        down_block_types=("DownEncoderBlock2D",) * num_blocks,
        up_block_types=("UpDecoderBlock2D",) * num_blocks,
        block_out_channels=block_out_channels,
        layers_per_block=1,
        act_fn="silu",
        latent_channels=latent_channels,
        norm_num_groups=32,
        sample_size=input_size,
        scaling_factor=1.0,
        force_upcast=True,
    )
    wrapper = AutoencoderKLWrapper(autoencoder)
    if device is not None:
        wrapper = wrapper.to(device)
    return wrapper


class DiTTransformer2DWrapper(nn.Module):
    def __init__(self, model: DiTTransformer2DModel):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, extra=None):
        if extra is None:
            extra = {}

        conditioning = extra.get("concat_conditioning")
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=1)

        class_labels = extra.get("label")
        if class_labels is None:
            class_labels = torch.zeros(
                x.shape[0],
                dtype=torch.long,
                device=x.device,
            )

        return self.model(
            hidden_states=x,
            timestep=timesteps,
            class_labels=class_labels,
            return_dict=True,
        ).sample

    def save_pretrained(
            self,
            save_directory,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            args=None,
            epoch=None,
            ema=None,
            **kwargs,
    ):
        save_directory = Path(save_directory)
        self.model.save_pretrained(save_directory, **kwargs)

        training_state = {}
        if epoch is not None:
            training_state["epoch"] = epoch
        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            training_state["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None:
            training_state["amp_scaler"] = scaler.state_dict()
        if args is not None:
            training_state["args"] = vars(args)
        if ema is not None:
            ema.save_pretrained(save_directory / EMA_DIR_NAME)
        if training_state:
            torch.save(training_state, save_directory / TRAINING_STATE_NAME)

    @classmethod
    def from_pretrained(
            cls,
            save_directory,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            device=None,
            return_training_state=False,
            ema=None,
            use_ema=False,
            **kwargs,
    ):
        save_directory = Path(save_directory)
        model = DiTTransformer2DModel.from_pretrained(
            save_directory,
            local_files_only=True,
            **kwargs,
        )
        if device is not None:
            model = model.to(device)
        wrapper = cls(model)
        if not return_training_state:
            return wrapper

        training_state_path = save_directory / TRAINING_STATE_NAME
        if not training_state_path.is_file():
            raise FileNotFoundError(
                f"Training state file not found: {training_state_path}"
            )
        map_location = device if device is not None else "cpu"
        try:
            training_state = torch.load(
                training_state_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            training_state = torch.load(
                training_state_path,
                map_location=map_location,
            )

        if optimizer is not None and training_state.get("optimizer"):
            optimizer.load_state_dict(training_state["optimizer"])
        if lr_scheduler is not None and training_state.get("lr_scheduler"):
            lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        if scaler is not None and training_state.get("amp_scaler"):
            scaler.load_state_dict(training_state["amp_scaler"])

        ema_path = save_directory / EMA_DIR_NAME
        if (ema is not None or use_ema) and ema_path.is_dir():
            ema_model = getattr(wrapper, "model", wrapper)
            loaded_ema = EMA.from_pretrained(
                ema_path,
                model_cls=type(ema_model),
            )
            if ema is not None:
                ema.load_state_dict(loaded_ema.state_dict())
            if use_ema:
                loaded_ema.copy_to(ema_model.parameters())

        return wrapper, int(training_state.get("epoch", 0)), training_state


def build_dit_transformer_2d_wrapper(
        model_arch,
        in_channels,
        sample_size,
        out_channels=None,
        num_embeds_ada_norm=1000,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu-approximate",
        upcast_attention=False,
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        device=None,
):
    if model_arch not in DIT_TRANSFORMER_2D_CONFIGS:
        supported = ", ".join(sorted(DIT_TRANSFORMER_2D_CONFIGS))
        raise ValueError(
            f"Unsupported DiT architecture {model_arch!r}. "
            f"Supported architectures: {supported}."
        )

    architecture = DIT_TRANSFORMER_2D_CONFIGS[model_arch]
    num_attention_heads = architecture["num_attention_heads"]

    model = DiTTransformer2DModel(
        num_attention_heads=num_attention_heads,
        attention_head_dim=architecture["attention_head_dim"],
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=architecture["num_layers"],
        dropout=dropout,
        attention_bias=attention_bias,
        sample_size=sample_size,
        patch_size=architecture["patch_size"],
        activation_fn=activation_fn,
        num_embeds_ada_norm=num_embeds_ada_norm,
        upcast_attention=upcast_attention,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=norm_elementwise_affine,
        norm_eps=norm_eps,
    )
    wrapper = DiTTransformer2DWrapper(model)
    if device is not None:
        wrapper = wrapper.to(device)
    return wrapper


class PixelDiT2DWrapper(nn.Module):
    """Wrapper that exposes a common training/checkpoint interface for ``PixDiT``."""

    def __init__(self, model: PixDiT):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, extra=None):
        if extra is None:
            extra = {}

        conditioning = extra.get("concat_conditioning")
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=1)

        labels = extra.get("label")
        if labels is None:
            labels = torch.zeros(
                x.shape[0],
                dtype=torch.long,
                device=x.device,
            )

        output = self.model(
            x,
            timesteps,
            labels,
            s=extra.get("s"),
            mask=extra.get("mask"),
        )

        return output

    def save_pretrained(
            self,
            save_directory,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            args=None,
            epoch=None,
            ema=None,
            **kwargs,
    ):
        save_directory = Path(save_directory)
        self.model.save_pretrained(save_directory, **kwargs)

        training_state = {}
        if epoch is not None:
            training_state["epoch"] = epoch
        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            training_state["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None:
            training_state["amp_scaler"] = scaler.state_dict()
        if args is not None:
            training_state["args"] = vars(args)
        if ema is not None:
            ema.save_pretrained(save_directory / EMA_DIR_NAME)
        if training_state:
            torch.save(training_state, save_directory / TRAINING_STATE_NAME)

    @classmethod
    def from_pretrained(
            cls,
            save_directory,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            device=None,
            return_training_state=False,
            ema=None,
            use_ema=False,
            **kwargs,
    ):
        save_directory = Path(save_directory)
        model = PixDiT.from_pretrained(
            save_directory,
            local_files_only=True,
            **kwargs,
        )
        if device is not None:
            model = model.to(device)
        wrapper = cls(model)

        ema_path = save_directory / EMA_DIR_NAME
        if (ema is not None or use_ema) and ema_path.is_dir():
            ema_model = getattr(wrapper, "model", wrapper)
            loaded_ema = EMA.from_pretrained(
                ema_path,
                model_cls=type(ema_model),
            )
            if ema is not None:
                ema.load_state_dict(loaded_ema.state_dict())
            if use_ema:
                loaded_ema.copy_to(ema_model.parameters())

        if not return_training_state:
            return wrapper

        training_state_path = save_directory / TRAINING_STATE_NAME
        if not training_state_path.is_file():
            raise FileNotFoundError(
                f"Training state file not found: {training_state_path}"
            )

        map_location = device if device is not None else "cpu"
        try:
            training_state = torch.load(
                training_state_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            training_state = torch.load(
                training_state_path,
                map_location=map_location,
            )

        if optimizer is not None and training_state.get("optimizer"):
            optimizer.load_state_dict(training_state["optimizer"])
        if lr_scheduler is not None and training_state.get("lr_scheduler"):
            lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        if scaler is not None and training_state.get("amp_scaler"):
            scaler.load_state_dict(training_state["amp_scaler"])

        return wrapper, int(training_state.get("epoch", 0)), training_state


def build_pixeldit_2d_wrapper(
        model_arch="PixelDiT_XL",
        in_channels=4,
        out_channels=None,
        num_groups=None,
        hidden_size=None,
        pixel_hidden_size=16,
        patch_depth=None,
        pixel_depth=None,
        patch_size=16,
        num_classes=1000,
        use_pixel_abs_pos=True,
        pit_adaln_post_modulation=False,
        upcast_attention=False,
        device=None,
):
    """Build a ``PixDiT`` model from a named architecture preset.

    The preset supplies the patch-level transformer width/depth. Explicit
    values for those arguments override the preset when provided.
    """
    if model_arch not in Pixel_DiT_2D_CONFIGS:
        supported = ", ".join(sorted(Pixel_DiT_2D_CONFIGS))
        raise ValueError(
            f"Unsupported PixelDiT architecture {model_arch!r}. "
            f"Supported architectures: {supported}."
        )

    architecture = Pixel_DiT_2D_CONFIGS[model_arch]
    num_groups = architecture["num_groups"] if num_groups is None else num_groups
    hidden_size = architecture["hidden_size"] if hidden_size is None else hidden_size
    patch_depth = architecture["patch_depth"] if patch_depth is None else patch_depth
    pixel_depth = architecture["pixel_depth"] if pixel_depth is None else pixel_depth

    model = PixDiT(
        in_channels=in_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        hidden_size=hidden_size,
        pixel_hidden_size=pixel_hidden_size,
        patch_depth=patch_depth,
        pixel_depth=pixel_depth,
        patch_size=patch_size,
        num_classes=num_classes,
        use_pixel_abs_pos=use_pixel_abs_pos,
        pit_adaln_post_modulation=pit_adaln_post_modulation,
        upcast_attention=upcast_attention,
    )
    wrapper = PixelDiT2DWrapper(model)
    if device is not None:
        wrapper = wrapper.to(device)
    return wrapper


if __name__ == "__main__":
    # Print parameter counts for every configured architecture.  Meta tensors
    # avoid allocating the large model weights just for this inspection.
    print("DiTTransformer2D configurations:")
    with torch.device("meta"):
        for model_arch in DIT_TRANSFORMER_2D_CONFIGS:
            model = build_dit_transformer_2d_wrapper(
                model_arch=model_arch,
                in_channels=4,
                out_channels=4,
                sample_size=32,
                num_embeds_ada_norm=1000,
                device="meta",
            )
            total_params, trainable_params, frozen_params = count_model_parameters(model)
            print(
                f"  {model_arch}: total={total_params:,}, "
                f"trainable={trainable_params:,}, frozen={frozen_params:,}"
            )

    print("PixelDiT configurations:")
    with torch.device("meta"):
        for model_arch in Pixel_DiT_2D_CONFIGS:
            model = build_pixeldit_2d_wrapper(
                model_arch=model_arch,
                in_channels=4,
                out_channels=4,
                device="meta",
            )
            total_params, trainable_params, frozen_params = count_model_parameters(model)
            print(
                f"  {model_arch}: total={total_params:,}, "
                f"trainable={trainable_params:,}, frozen={frozen_params:,}"
            )

    x = torch.randn(2, 3, 32, 32)
    timesteps = torch.randint(0, 1000, (2,))

    model = build_dit_transformer_2d_wrapper(
        model_arch="DiT_T_4",
        in_channels=3,
        out_channels=3,
        sample_size=32,
        num_embeds_ada_norm=10,
    )

    output = model(x, timesteps)
    print("Unconditional output:", output.shape)

    extra = {"label": torch.tensor([1, 2])}
    output = model(x, timesteps, extra)
    print("Class-conditional output:", output.shape)

    model = build_dit_transformer_2d_wrapper(
        model_arch="DiT_T_4",
        in_channels=5,
        out_channels=3,
        sample_size=32,
        num_embeds_ada_norm=10,
    )
    extra = {
        "concat_conditioning": torch.randn(2, 2, 32, 32),
    }
    output = model(x, timesteps, extra)
    print("Concat conditioning output:", output.shape)

    model = build_pixeldit_2d_wrapper(
        model_arch="PixelDiT_XL",
        in_channels=3,
        out_channels=3,
    )
    output = model(x, timesteps)
    print("PixelDiT output:", output.shape)
