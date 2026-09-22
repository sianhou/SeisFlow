from pathlib import Path
import subprocess
import sys

import torch

from models.pixeldit import TimestepConditioner
from models.unet2 import UNet2DModel
from models.wrapper import UNet2DWrapper, build_unet_2d_wrapper


def build_tiny_unet(**kwargs):
    config = dict(
        in_channels=3,
        out_channels=1,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
    )
    return UNet2DModel(**{**config, **kwargs})


def test_continuous_time_matches_dit_conditioner():
    model = build_tiny_unet(max_period=37, frequency_embedding_size=64)
    reference = TimestepConditioner(128, frequency_embedding_size=64, max_period=37)
    reference.load_state_dict(model.unet.time_embed.state_dict())
    time = torch.tensor([0.0, 0.375, 1.0])
    expected = reference(time)
    captured = []
    handle = model.unet.time_embed.register_forward_hook(
        lambda module, inputs, output: captured.append((inputs[0], output))
    )
    model(torch.randn(3, 3, 8, 12), time)
    handle.remove()
    torch.testing.assert_close(captured[0][0], time)
    torch.testing.assert_close(captured[0][1], expected)


def test_rectangular_forward_and_backward():
    model = build_tiny_unet()
    output = model(torch.randn(2, 3, 8, 12), torch.rand(2))
    assert output.shape == (2, 1, 8, 12)
    assert torch.count_nonzero(output) == 0
    (output - torch.randn_like(output)).square().mean().backward()
    assert model.unet.out[-1].weight.grad.abs().sum() > 0


def test_wrapper_passes_spatial_and_class_conditions():
    model = build_tiny_unet(num_classes=3).eval()
    # Make predictions nonzero so equality does not just test the zero head.
    torch.nn.init.normal_(model.unet.out[-1].weight, std=0.02)
    wrapper = UNet2DWrapper(model)
    x, conditioning = torch.randn(2, 1, 8, 12), torch.randn(2, 2, 8, 12)
    time, labels = torch.rand(2), torch.tensor([1, 2])
    actual = wrapper(x, time, {"concat_conditioning": conditioning, "label": labels})
    expected = model(torch.cat((x, conditioning), dim=1), time, labels)
    torch.testing.assert_close(actual, expected)


def test_checkpoint_roundtrip_preserves_time_settings_and_predictions(tmp_path):
    model = build_tiny_unet(max_period=25, frequency_embedding_size=64).eval()
    torch.nn.init.normal_(model.unet.out[-1].weight, std=0.02)
    wrapper = UNet2DWrapper(model)
    wrapper.save_pretrained(tmp_path, epoch=7)
    restored, epoch, _ = UNet2DWrapper.from_pretrained(
        tmp_path, return_training_state=True,
    )
    restored.eval()
    assert epoch == 7
    assert restored.model.config.max_period == 25
    assert restored.model.config.frequency_embedding_size == 64
    x, time = torch.randn(2, 3, 8, 12), torch.rand(2)
    torch.testing.assert_close(wrapper(x, time), restored(x, time))


def test_builder_defaults_and_overrides():
    wrapper = build_unet_2d_wrapper("Nano", max_period=20, device="cpu")
    assert wrapper.model.config.model_channels == 32
    assert wrapper.model.config.num_res_blocks == 1
    assert wrapper.model.config.frequency_embedding_size == 256
    assert wrapper.model.unet.time_embed.max_period == 20


def test_unet2_runs_without_other_model_modules():
    # A fresh interpreter ensures previous imports cannot hide dependencies.
    script = """
import sys
sys.modules.update({name: None for name in (
    'models.unet', 'models.nn', 'models.pixeldit',
)})
import torch
from models.unet2 import UNet2DModel
torch.set_num_threads(1)
model = UNet2DModel(model_channels=32, num_res_blocks=1, channel_mult=(1, 2))
output = model(torch.randn(2, 3, 8, 12), torch.tensor([0.0, 1.0]))
assert output.shape == (2, 1, 8, 12)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
