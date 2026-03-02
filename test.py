import argparse
import logging
import sys
import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel
from tools.seed_everything import seed_everything
from tools.segy_dataset import SegyDataset, ClipFirstChannel, SliceLastDim

logger = logging.getLogger(__name__)


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


MODEL_CONFIGS = {
    "simple": {
        "in_channels": 1,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],  # 64x64 和 32x32
        "dropout": 0.2,
        "channel_mult": [1, 2, 4, 8],  # 256→128→64→32
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": True,
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def create_parser():
    parser = argparse.ArgumentParser(description="PyTorch U-Net Training")


batch_size = 32
seed = 42
lr = 0.0001
optimizer_betas = (0.9, 0.95)


def train():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)

    logger.info(f"Initializing Dataset:")
    transform = transforms.Compose([
        SliceLastDim(0, 1501),
        ClipFirstChannel(-2, 2),
        transforms.Resize((256, 256)),
    ])
    dataset = SegyDataset("ma2+GathAP.sgy", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=True)

    logger.info("Initializing Model")
    model = UNetModel(**MODEL_CONFIGS["simple"])
    total, trainable, frozen = count_params(model)
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Trainable params: {trainable:,}")
    logger.info(f"Frozen params:    {frozen:,}")
    model.to(device)

    logger.info(f"Learning rate: {lr:.2e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    start_time = time.time()
    logger.info(f"Star training: {start_time}")
    for epoch in range(1000):

        model.train()
        epoch_loss = 0.0

        path = CondOTProbPath()
        for data_iter_step, (samples) in enumerate(dataloader):
            samples = samples[:, 0].unsqueeze(1).to(device)
            samples = samples * 0.5  # [-1, 1]

            noise = torch.randn_like(samples).to(device)
            t = torch.torch.rand(samples.shape[0]).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            # ===== 前向 =====
            pred = model(x_t, t, extra={})
            loss = F.mse_loss(pred, u_t)

            # ===== 反向 =====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:04d} | "
            f"Loss {epoch_loss / len(dataloader):.6f} | "
            f"Time {epoch_time:.2f}s"
        )

        # ===== 每100个epoch保存模型 =====
        if (epoch + 1) % 100 == 0:
            save_path = f"model_epoch_{epoch + 1:04d}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
            self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = False
        assert (
                cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval(checkpoint):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetModel(**MODEL_CONFIGS["simple"])
    model.load_state_dict(torch.load(checkpoint))
    # model.eval()

    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.to(device)
    cfg_scaled_model.train(False)

    solver = ODESolver(velocity_model=cfg_scaled_model)

    x_0 = torch.randn([4, 1, 256, 256], dtype=torch.float32, device=device)
    time_grid = torch.tensor([0.0, 1.0], device=device)

    synthetic_samples = solver.sample(
        time_grid=time_grid,
        x_init=x_0,
        return_intermediates=False,
        step_size=0.01,
        cfg_scale=0.0,
        label=None,
    )

    print(synthetic_samples.shape)
    imgs = synthetic_samples[:, 0]  # [B,H,W]
    imgs = imgs.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    for i in range(4):
        axes[i].imshow(imgs[i].T, cmap="seismic", origin="upper", vmin=-1, vmax=1)
        axes[i].set_title(f"sample {i} (ch0)")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"sample_{checkpoint}.png")


if __name__ == '__main__':
    train()
    # for epoch in range(1000):
    #     if (epoch + 1) % 100 == 0:
    #         save_path = f"model_epoch_{epoch + 1:04d}.pth"
    #         eval(save_path)
