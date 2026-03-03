import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchvision import transforms

from core.count_params import count_params
from core.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from core.logger import BaseLogger
from core.seed_everything import seed_everything
from core.segy_dataset import SliceLastDim, ClipFirstChannel, ScaleFirstChannel, SegyDataset
from flow_matching.path import CondOTProbPath
from models.unet import UNetModel
from training import distributed_mode

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
    parser = argparse.ArgumentParser(description='Synthetic seismic dataset training')
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * gpus")
    parser.add_argument("--dataset", default="/disk03/hsa/SeisFlow/ma2+GathAP.sgy",
                        help="Segy data for training / testing")
    parser.add_argument("--decay_lr", action="store_true", help="Adds a linear decay to the lr during training.")
    parser.add_argument("--device", default="cuda", help="Device to use for training / testing")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (absolute lr)")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--optimizer_betas", nargs="+", type=float, default=[0.9, 0.95],
                        help="learning rate (absolute lr)")
    parser.add_argument("--output_dir", default="./output_dir", help="Path where to save, empty for no saving", )
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    return parser


def main(args):
    distributed_mode.init_distributed_mode(args)

    logger = BaseLogger(log_dir=args.output_dir, overwrite=True)
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))

    # device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    seed_everything(seed)

    # dataset
    logger.info(f"Initializing Dataset: {args.dataset}")
    transform = transforms.Compose([
        SliceLastDim(0, 1501),
        ClipFirstChannel(-2, 2),
        ScaleFirstChannel(0.5),
        transforms.Resize((256, 256)),
    ])
    dataset_train = SegyDataset(args.dataset, transform=transform)

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
    model = UNetModel(**MODEL_CONFIGS["simple"])

    model.to(device)
    model_without_ddp = model
    total, trainable, frozen = count_params(model_without_ddp)
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

    loss_scaler = NativeScaler()

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

            noise = torch.randn_like(samples).to(device)

            t = torch.torch.rand(samples.shape[0]).to(device)

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            pred = model(x_t, t, extra={})

            with torch.cuda.amp.autocast():
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
        if (epoch + 1) % 100 == 0 and distributed_mode.get_rank() == 0:
            save_path = f"model_epoch_{epoch + 1:04d}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
