import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import MeanMetric
from torchvision import transforms

from core.dataset import PatchDataset, SegyDataset
from core.logging.logger import SimpleLogger
from core.masks.row_mask import generate_random_row_mask
from core.metrics import compute_psnr
from core.training import set_random_seed, count_model_parameters, AMPGradScaler
from core.transforms import PerChannelMinMaxToMinusOneOne, SliceLastDimension, ClipFirstChannel, ScaleFirstChannel
from core.visualization import plot_seismic_grid
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.dit import DiT
from models.unet import UNetModel
from training import distributed_mode

MODEL_CONFIGS = {
    "simple_unet": {
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
    "simple_dit": {
        "in_channels": 3,
        "out_channels": 1,
        "input_size": 32,
        "patch_size": 4,
        "hidden_size": 384,
        "depth": 4,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "num_classes": None,
        "class_dropout_prob": 0.1,
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


def create_parser():
    parser = argparse.ArgumentParser(description='Synthetic seismic dataset training')
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * gpus")
    parser.add_argument("--dataset", default="./dataset_train",
                        help="Segy data for training / testing")
    parser.add_argument("--decay_lr", action="store_true", help="Adds a linear decay to the lr during training.")
    parser.add_argument("--device", default="cuda", help="Device to use for training / testing")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (absolute lr)")
    parser.add_argument("--missing_ratio", nargs="+", type=float, default=[0.3, 0.7])
    parser.add_argument("--mode", default="train", choices=["train", "sample"])
    parser.add_argument("--model_path", default="/disk03/hsa/SeisFlow/model_epoch_0100.pth",
                        help="Pre-trained model for sampling")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--save_frequency", default=50, type=int)
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--optimizer_betas", nargs="+", type=float, default=[0.9, 0.95],
                        help="learning rate (absolute lr)")
    parser.add_argument("--output_dir", default="./output_dir", help="Path where to save, empty for no saving", )
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    return parser


def train(args):
    distributed_mode.init_distributed_mode(args)

    logger = SimpleLogger(log_dir=args.output_dir, overwrite=True)
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))

    # device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    set_random_seed(seed)

    # dataset
    logger.info(f"Initializing Dataset: {args.dataset}")
    transform = transforms.Compose([
        PerChannelMinMaxToMinusOneOne()
    ])
    dataset_train = PatchDataset(args.dataset, transform=transform)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    # model
    logger.info("Initializing Model")
    model = UNetModel(**MODEL_CONFIGS["simple_unet"])

    model.to(device)
    model_without_ddp = model
    total, trainable, frozen = count_model_parameters(model_without_ddp)
    logger.info(str(model_without_ddp))
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Trainable params: {trainable:,}")
    logger.info(f"Frozen params:    {frozen:,}")

    eff_batch_size = (
            args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # Optimizer
    logger.info("Initializing Optimizer")
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )

    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = AMPGradScaler()

    # load_model(
    #     args=args,
    #     model_without_ddp=model_without_ddp,
    #     optimizer=optimizer,
    #     loss_scaler=loss_scaler,
    #     lr_schedule=lr_schedule,
    # )

    logger.info(f"Start training:")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train_one_epoch
        gc.collect()
        model.train(True)
        batch_loss = MeanMetric().to(device, non_blocking=True)
        epoch_loss = MeanMetric().to(device, non_blocking=True)

        accum_iter = args.accum_iter

        path = CondOTProbPath()

        for data_iter_step, (samples) in enumerate(data_loader_train):
            if data_iter_step % accum_iter == 0:
                optimizer.zero_grad()
                batch_loss.reset()

            samples = samples[:, 0].unsqueeze(1).to(device)
            missing_ratio = np.random.uniform(args.missing_ratio[0], args.missing_ratio[1])
            mask = generate_random_row_mask(x=samples, missing_ratio=missing_ratio)
            missed = mask * samples
            extra = {}
            extra["concat_conditioning"] = torch.cat((missed, mask), dim=1)

            noise = torch.randn_like(samples).to(device)

            t = torch.torch.rand(samples.shape[0]).to(device)

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.amp.autocast(device_type="cuda"):
                pred = model(x_t, t, extra=extra)
                loss = F.mse_loss(pred, u_t)

            loss_value = loss.item()
            batch_loss.update(loss)
            epoch_loss.update(loss)

            loss /= accum_iter

            apply_update = (data_iter_step + 1) % accum_iter == 0
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=apply_update,
            )

            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader_train)}]: loss = {batch_loss.compute()}, lr = {lr}")

        lr_schedule.step()

        # ===== 每100个epoch保存模型 =====
        if (epoch + 1) % args.save_frequency == 0 and distributed_mode.get_rank() == 0:
            save_path = f"{args.output_dir}/model_epoch_{epoch + 1:05d}.pth"
            torch.save(model_without_ddp.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

            if args.save_samples:
                raw = samples[0:4].detach().cpu().numpy()
                noised = x_t[0:4].detach().cpu().numpy()
                mask = mask[0:4].detach().cpu().numpy()
                missed = missed[0:4].detach().cpu().numpy()

                plot_seismic_grid(raw, f"{args.output_dir}/raw_epoch{epoch}_rank{distributed_mode.get_rank()}.png",
                                  title="raw")
                plot_seismic_grid(noised,
                                  f"{args.output_dir}/noised_epoch{epoch}_rank{distributed_mode.get_rank()}.png",
                                  title="noised")
                plot_seismic_grid(mask, f"{args.output_dir}/mask_epoch{epoch}_rank{distributed_mode.get_rank()}.png",
                                  title="mask")
                plot_seismic_grid(missed,
                                  f"{args.output_dir}/missed_epoch{epoch}_rank{distributed_mode.get_rank()}.png",
                                  title="missed")

    if args.distributed:
        distributed_mode.barrier()
        distributed_mode.destroy()


def sample(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    set_random_seed(args.seed)

    # data
    transform = transforms.Compose([
        SliceLastDimension(0, 1501),
        ClipFirstChannel(-2, 2),
        ScaleFirstChannel(0.5),
        transforms.Resize((256, 256)),
    ])
    dataset = SegyDataset("ma2+GathAP.sgy", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=True)
    samples = next(iter(dataloader))
    samples = samples[:, 0].unsqueeze(1).to(device)
    missing_ratio = np.random.uniform(args.missing_ratio[0], args.missing_ratio[1])
    mask = generate_random_row_mask(x=samples, missing_ratio=missing_ratio)
    missed = mask * samples
    extra = {}
    extra["concat_conditioning"] = torch.cat((missed, mask), dim=1)

    # model
    model = DiT(**MODEL_CONFIGS["simple_dit"])
    model.load_state_dict(torch.load(args.model_path))

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
        concat_conditioning=extra
    )

    synthetic_samples = missed + (1.0 - mask) * synthetic_samples

    print(f"seed: {args.seed}")
    print(f"missing_ratio: {missing_ratio}")
    print(f"model: {args.model_path}")

    # plot
    raw = samples[0:4].detach().cpu().numpy()
    mask = mask[0:4].detach().cpu().numpy()
    missed = missed[0:4].detach().cpu().numpy()
    recon = synthetic_samples[0:4].detach().cpu().numpy()

    plot_seismic_grid(raw, f"{args.output_dir}/raw.png", title="raw")
    plot_seismic_grid(mask, f"{args.output_dir}/mask.png", title="mask")
    plot_seismic_grid(missed, f"{args.output_dir}/missed.png", title="missed")
    plot_seismic_grid(recon, f"{args.output_dir}/recon.png", title="missed")
    plot_seismic_grid(raw - recon, f"{args.output_dir}/diff.png", title="missed")

    for i in range(4):
        psnr = compute_psnr(raw[i], recon[i], 1.0)
        print(f"psnr[{i}]: {psnr}]")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    else:
        exit()
