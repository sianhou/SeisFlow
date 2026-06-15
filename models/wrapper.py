import math
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL, DiTTransformer2DModel
from torch import nn

TRAINING_STATE_NAME = "training_state.pth"

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

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, save_directory, device=None, **kwargs):
        model = DiTTransformer2DModel.from_pretrained(
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


def build_dit_transformer_2d_wrapper(
        model_arch,
        in_channels,
        sample_size,
        out_channels=None,
        num_embeds_ada_norm=1,
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


if __name__ == "__main__":
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
