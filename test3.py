# device
import numpy as np
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from core.generate_mask import generate_random_row_mask
from core.plot_seismic import plot_one_row
from core.seed_everything import seed_everything
from core.segy_dataset import SliceLastDim, ClipFirstChannel, ScaleFirstChannel, SegyDataset
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel

MODEL_CONFIGS = {
    "simple": {
        "in_channels": 3,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],  # 64x64 和 32x32
        "dropout": 0.2,
        "channel_mult": [1, 2, 4, 8],  # 256→128→64→32
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
            self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor,
            concat_conditioning
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
                result = self.model(x, t, extra=concat_conditioning)

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def generate_time_intervals(breakpoints):
    """
    Generate time intervals from a list of breakpoints.

    Args:
        breakpoints (list): List of time points [t0, t1, t2, ..., tn]

    Returns:
        list: List of tuples containing consecutive time intervals
    """
    return [(float(breakpoints[i]), float(breakpoints[i + 1])) for i in range(len(breakpoints) - 1)]


device = torch.device("cuda")

# fix the seed for reproducibility
seed_everything(42)

# data
transform = transforms.Compose([
    SliceLastDim(0, 1501),
    ClipFirstChannel(-2, 2),
    ScaleFirstChannel(0.5),
    transforms.Resize((256, 256)),
])
dataset = SegyDataset("ma2+GathAP.sgy", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=0,
                                         pin_memory=True,
                                         drop_last=True)
samples = next(iter(dataloader))
samples = samples[:, 0].unsqueeze(1).to(device)
mask = generate_random_row_mask(x=samples, missing_ratio=0.3)
missed = mask * samples
extra = {}
extra["concat_conditioning"] = torch.cat((missed, mask), dim=1)

# model
model = UNetModel(**MODEL_CONFIGS["simple"])
model.load_state_dict(torch.load("./output_dir/model_epoch_00200.pth"))

cfg_scaled_model = CFGScaledModel(model=model)
cfg_scaled_model.to(device)
cfg_scaled_model.train(False)

solver = ODESolver(velocity_model=cfg_scaled_model)

x_0 = torch.randn([1, 1, 256, 256], dtype=torch.float32, device=device)

time_grid = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device)

synthetic_samples = solver.sample(
    time_grid=time_grid,
    x_init=x_0,
    return_intermediates=True,
    step_size=0.01,
    cfg_scale=0.0,
    label=None,
    concat_conditioning=extra
).squeeze()

imgs = synthetic_samples.detach().cpu().numpy()
plot_one_row(
    imgs, "synthetic_samples_one_step.png", "synthetic_samples_one_step", -1, 1)

time_intervals = generate_time_intervals(np.round(np.arange(0, 1.001, 0.2), 2))

synthetic_samples = [x_0]
path_samples = [samples]

current_x = x_0

path = CondOTProbPath()
for start, end in time_intervals:
    # calculate x_t-1
    time_grid_interval = torch.tensor([start, end], device=device)
    current_x = solver.sample(
        time_grid=time_grid_interval,
        x_init=current_x,
        return_intermediates=False,
        step_size=0.01,
        cfg_scale=0.0,
        label=None,
        concat_conditioning=extra
    )

    # sample x_t-1 from x_0
    noise = torch.randn_like(current_x).to(device)

    t = torch.torch.zeros(current_x.shape[0]).fill_(end).to(device)

    path_sample = path.sample(t=t, x_0=noise, x_1=samples)

    current_x = path_sample.x_t * mask + current_x * (1 - mask)

    synthetic_samples.append(current_x)

synthetic_samples_multi_step = torch.cat(synthetic_samples, axis=0)

imgs2 = synthetic_samples_multi_step.squeeze().detach().cpu().numpy()
plot_one_row(
    imgs2, "synthetic_samples_multi_step2.png", "synthetic_samples_multi_step2", -1, 1)

plot_one_row(
    imgs - imgs2, "diff.png", "diff", -1, 1)
