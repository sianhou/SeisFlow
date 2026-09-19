from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.trainer import Trainer
from core.training.amp_scaler import AMPGradScaler


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, x, timesteps, extra=None):
        del timesteps, extra
        return self.weight * x


class AccumulationTrainer(Trainer):
    def setup_dataset(self):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def sample_path(self, x1, mode="linear"):
        del mode
        return {
            "x_t": x1,
            "t": torch.zeros(x1.shape[0]),
        }

    def compute_loss(self, prediction, sample, mode="velocity"):
        del sample, mode
        loss = prediction.mean()
        return loss, loss, 0


class RecordingSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class RecordingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class LifecycleTrainer(AccumulationTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.trained_epochs = []
        self.saved_epochs = []

    def train_one_epoch(self, epoch):
        self.trained_epochs.append(epoch)
        return float(epoch)

    def save_pretrained(self, epoch):
        self.saved_epochs.append(epoch)


class SkippingScaler:
    optimizer_step_was_skipped = True

    def __call__(self, loss, optimizer, **kwargs):
        del optimizer, kwargs
        loss.backward()
        return torch.zeros(())


class OverflowGradScaler:
    def __init__(self):
        self.scale_value = 8.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        del optimizer

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        del optimizer

    def update(self):
        self.scale_value /= 2.0


@pytest.mark.parametrize("total_steps", [8, 6])
def test_gradient_accumulation_uses_one_fixed_group_size(total_steps):
    trainer = AccumulationTrainer(
        SimpleNamespace(
            grad_accum_steps=4,
            clip_grad=0.0,
        )
    )
    trainer.device = torch.device("cpu")
    trainer.model = ScalarModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.scaler = AMPGradScaler(enabled=False, device="cpu")
    trainer.dataloader = [torch.ones(1) for _ in range(total_steps)]

    trainer.train_one_epoch(epoch=0)

    # Every accumulation group contains identical unit gradients, so each
    # optimizer step must contribute exactly one, including a partial tail.
    expected_optimizer_steps = (total_steps + 3) // 4
    assert trainer.model.weight.item() == pytest.approx(-expected_optimizer_steps)


@pytest.mark.parametrize(
    ("num_epochs", "expected_saved_epochs"),
    [
        (3, [2, 3]),
        (4, [2, 4]),
    ],
)
def test_train_sets_sampler_epoch_and_saves_final_epoch(
        tmp_path,
        num_epochs,
        expected_saved_epochs,
):
    trainer = LifecycleTrainer(
        SimpleNamespace(
            distributed=True,
            num_epochs=num_epochs,
            save_every_epochs=2,
        )
    )
    sampler = RecordingSampler()
    trainer.dataloader = SimpleNamespace(sampler=sampler)
    trainer.lr_scheduler = RecordingScheduler()
    trainer.logger = SimpleNamespace(run_dir=tmp_path, log_event=lambda *a, **k: None)

    trainer.train()

    assert sampler.epochs == list(range(num_epochs))
    assert trainer.trained_epochs == list(range(num_epochs))
    assert trainer.lr_scheduler.steps == num_epochs
    assert trainer.saved_epochs == expected_saved_epochs


def test_ema_is_not_updated_when_amp_skips_optimizer_step():
    trainer = AccumulationTrainer(
        SimpleNamespace(
            grad_accum_steps=1,
            clip_grad=0.0,
        )
    )
    trainer.device = torch.device("cpu")
    trainer.model = ScalarModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.scaler = SkippingScaler()
    trainer.dataloader = [torch.ones(1)]
    ema_updates = []
    trainer.update_ema = lambda: ema_updates.append(True)

    trainer.train_one_epoch(epoch=0)

    assert ema_updates == []
    assert trainer.model.weight.item() == 0.0


def test_amp_scaler_reports_an_overflow_skipped_step():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scaler = AMPGradScaler(enabled=False, device="cpu")
    scaler.enabled = True
    scaler._scaler = OverflowGradScaler()

    scaler(
        model.weight,
        optimizer,
        parameters=model.parameters(),
        update_grad=True,
    )

    assert scaler.optimizer_step_was_skipped is True
    assert model.weight.item() == 0.0
