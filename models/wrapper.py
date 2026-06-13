import math
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from torch import nn

TRAINING_STATE_NAME = "training_state.pth"


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
