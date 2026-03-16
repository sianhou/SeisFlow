import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchvision import transforms

from core.dataset import PatchDataset
from core.logging.logger import SimpleLogger
from core.masks.row_mask import generate_random_row_mask
from core.training import AMPGradScaler, count_model_parameters, set_random_seed
from core.transforms import PerChannelMinMaxToMinusOneOne
from flow_matching.path import CondOTProbPath
from models.dit import DiT
from models.unet import UNetModel
from training import distributed_mode

MODEL_CONFIGS = {
    "dit": {
        "in_channels": 3,
        "out_channels": 1,
        "input_size": 32,
        "patch_size": 4,
        "hidden_size": 384,
        "depth": 8,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "num_classes": None,
        "class_dropout_prob": 0.1,
    },
    "unet": {
        "in_channels": 3,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.2,
        "channel_mult": [1, 2, 4, 8],
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

MODEL_BUILDERS = {
    "dit": DiT,
    "unet": UNetModel,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train DiT on seismic patches")
    parser.add_argument(
        "--grad_accum_steps",
        default=1,
        type=int,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Mini-batch size per GPU.",
    )
    parser.add_argument(
        "--data_dir",
        default="./train_dataset",
        help="Training dataset directory.",
    )
    parser.add_argument(
        "--use_lr_decay",
        action="store_true",
        help="Apply linear learning-rate decay over training.",
    )
    parser.add_argument("--device", default="cuda", help="Training device.")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="URL used to initialize distributed training.",
    )
    parser.add_argument(
        "--num_epochs",
        default=1000,
        type=int,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--mask_ratio_range",
        nargs=2,
        type=float,
        default=[0.3, 0.7],
        help="Range of random row-mask ratios.",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--checkpoint_interval",
        default=50,
        type=int,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--adam_betas",
        nargs=2,
        type=float,
        default=[0.9, 0.95],
        help="Beta coefficients for AdamW.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Directory used for logs and checkpoints.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pinned memory in the dataloader.",
    )
    parser.add_argument(
        "--model_name",
        choices=sorted(MODEL_CONFIGS.keys()),
        default="dit",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of distributed processes.",
    )
    return parser


def build_dataloader(args):
    dataset = PatchDataset(
        args.data_dir,
        transform=transforms.Compose([PerChannelMinMaxToMinusOneOne()]),
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=distributed_mode.get_world_size(),
        rank=distributed_mode.get_rank(),
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    return dataset, sampler, dataloader


def build_model(model_name, device):
    model_class = MODEL_BUILDERS[model_name]
    model = model_class(**MODEL_CONFIGS[model_name]).to(device)
    return model


def train_one_epoch(model, dataloader, optimizer, scaler, flow_path, device, args, logger, epoch):
    gc.collect()
    model.train(True)

    running_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    total_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        if step % args.grad_accum_steps == 0:
            optimizer.zero_grad()
            running_loss.reset()

        clean_images = batch[:, 0].unsqueeze(1).to(device, non_blocking=True)
        mask_ratio = np.random.uniform(*args.mask_ratio_range)
        observed_mask = generate_random_row_mask(x=clean_images, missing_ratio=mask_ratio)
        masked_images = observed_mask * clean_images
        conditioning = torch.cat((masked_images, observed_mask), dim=1)

        noise = torch.randn_like(clean_images)
        timesteps = torch.rand(clean_images.shape[0], device=device)
        flow_sample = flow_path.sample(t=timesteps, x_0=noise, x_1=clean_images)
        noisy_images = flow_sample.x_t
        target_velocity = flow_sample.dx_t

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predicted_velocity = model(
                noisy_images,
                timesteps,
                extra={"concat_conditioning": conditioning},
            )
            loss = F.mse_loss(predicted_velocity, target_velocity)

        running_loss.update(loss)
        epoch_loss.update(loss)

        loss = loss / args.grad_accum_steps
        should_step = (step + 1) % args.grad_accum_steps == 0
        scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=should_step,
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1} [{step + 1}/{total_steps}] "
            f"loss={running_loss.compute():.6f}, lr={learning_rate:.2e}"
        )

    return epoch_loss.compute().item()


def save_checkpoint(model, output_dir, epoch):
    checkpoint_path = Path(output_dir) / f"model_epoch_{epoch + 1:05d}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def main(args):
    distributed_mode.init_distributed_mode(args)

    logger = SimpleLogger(log_dir=args.output_dir, overwrite=True)
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    set_random_seed(seed)

    logger.info(f"Initializing dataset: {args.data_dir}")
    _, train_sampler, train_loader = build_dataloader(args)
    logger.info(str(train_sampler))

    logger.info(f"Initializing model: {args.model_name}")
    model = build_model(args.model_name, device)
    model_without_ddp = model

    total_params, trainable_params, frozen_params = count_model_parameters(model_without_ddp)
    logger.info(str(model_without_ddp))
    logger.info(f"Total params:     {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    logger.info(f"Frozen params:    {frozen_params:,}")

    effective_batch_size = (
            args.batch_size * args.grad_accum_steps * distributed_mode.get_world_size()
    )
    logger.info(f"Learning rate: {args.learning_rate:.2e}")
    logger.info(f"Gradient accumulation steps: {args.grad_accum_steps}")
    logger.info(f"Effective batch size: {effective_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    logger.info("Initializing optimizer")
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.adam_betas),
    )

    if args.use_lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.num_epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.learning_rate,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            total_iters=args.num_epochs,
            factor=1.0,
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"LR scheduler: {lr_scheduler}")

    scaler = AMPGradScaler()
    flow_path = CondOTProbPath()

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(args.num_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            flow_path=flow_path,
            device=device,
            args=args,
            logger=logger,
            epoch=epoch,
        )
        lr_scheduler.step()
        logger.info(f"Epoch {epoch + 1} finished with mean loss {epoch_loss:.6f}")

        if (
                (epoch + 1) % args.checkpoint_interval == 0
                and distributed_mode.get_rank() == 0
        ):
            checkpoint_path = save_checkpoint(model_without_ddp, args.output_dir, epoch)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.2f}s")

    if args.distributed:
        distributed_mode.barrier()
        distributed_mode.destroy()


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
