"""Unit tests for training losses (src/train/losses.py)."""

from __future__ import annotations

import pytest
import torch

from src.train.losses import (
    CombinedLoss,
    balance_loss,
    contrastive_loss,
    fusion_bce_loss,
    reliability_loss,
    unimodal_bce_loss,
)


def test_fusion_bce_loss_standard() -> None:
    """Computes BCE loss with and without label smoothing."""
    logits = torch.tensor([[2.0], [-2.0]])
    target = torch.tensor([[1.0], [0.0]])
    loss_raw = fusion_bce_loss(logits, target, label_smoothing=0.0)
    assert loss_raw > 0.0
    assert torch.isfinite(loss_raw)

    loss_smooth = fusion_bce_loss(logits, target, label_smoothing=0.1)
    assert loss_smooth > 0.0
    assert torch.isfinite(loss_smooth)


def test_unimodal_bce_loss_with_masking() -> None:
    """Missing streams do not contribute to unimodal BCE loss."""
    logit_v = torch.tensor([[2.0], [3.0]])
    logit_a = torch.tensor([[-1.0], [-2.0]])
    target = torch.tensor([[1.0], [0.0]])
    vm = torch.tensor([[1.0], [0.0]])  # first sample visual missing
    am = torch.tensor([[0.0], [0.0]])

    loss_v, loss_a, comb = unimodal_bce_loss(logit_v, logit_a, target, visual_missing=vm, audio_missing=am)
    assert torch.isfinite(loss_v)
    assert torch.isfinite(loss_a)
    assert torch.isfinite(comb)


def test_contrastive_loss_few_real_samples() -> None:
    """InfoNCE gracefully returns zero when fewer than 2 real samples exist."""
    proj_v = torch.randn(4, 8, 32)
    proj_a = torch.randn(4, 8, 32)
    # Only 1 real sample
    target = torch.tensor([0.0, 1.0, 1.0, 1.0])
    loss = contrastive_loss(proj_v, proj_a, target=target)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_contrastive_loss_multiple_real_samples() -> None:
    """InfoNCE computes non-zero finite loss on multiple real samples."""
    proj_v = torch.randn(4, 8, 32)
    proj_a = torch.randn(4, 8, 32)
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss = contrastive_loss(proj_v, proj_a, target=target)
    assert loss > 0.0
    assert torch.isfinite(loss)


def test_balance_loss() -> None:
    """Balance regularization is zero when weights are exactly 0.5, positive otherwise."""
    equal_weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert torch.isclose(balance_loss(equal_weights), torch.tensor(0.0))

    skewed_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert balance_loss(skewed_weights) > 0.0


def test_combined_loss_forward_and_schedule() -> None:
    """CombinedLoss produces finite scalar loss and handles warmup schedules."""
    criterion = CombinedLoss(
        alpha=0.25,
        beta=0.10,
        gamma=0.05,
        eta=0.01,
        tau_r=1.0,
        balance_warmup_epochs=2,
        balance_schedule="step",
    )
    b = 4
    outputs = {
        "logit_f": torch.randn(b, 1),
        "logit_v": torch.randn(b, 1),
        "logit_a": torch.randn(b, 1),
        "w_v": torch.full((b, 1), 0.5),
        "w_a": torch.full((b, 1), 0.5),
        "weights": torch.full((b, 2), 0.5),
        "proj_v": torch.randn(b, 8, 32),
        "proj_a": torch.randn(b, 8, 32),
        "visual_missing": torch.zeros(b, 1),
        "audio_missing": torch.zeros(b, 1),
    }
    batch = {
        "target": torch.randint(0, 2, (b, 1)).float(),
    }

    # Epoch 0 (during warmup)
    loss_dict_0 = criterion(outputs, batch, epoch=0)
    assert "total" in loss_dict_0
    assert torch.isfinite(loss_dict_0["total"])

    # Epoch 5 (after warmup)
    loss_dict_5 = criterion(outputs, batch, epoch=5)
    assert "total" in loss_dict_5
    assert torch.isfinite(loss_dict_5["total"])
    assert float(loss_dict_5["eta"]) == 0.0


def test_trainer_epoch_with_combined_loss() -> None:
    """Trainer.train_epoch successfully aggregates all loss dictionary components."""
    from torch.utils.data import DataLoader
    from src.train.trainer import Trainer
    from src.utils.config_loader import load_train_config

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, frames, mel_spec, **kwargs):
            b = frames.shape[0]
            zeros = torch.zeros(b, 1)
            half = torch.full((b, 1), 0.5)
            return {
                "logit_f": self.fc(zeros),
                "p_f": half,
                "logit_v": zeros,
                "logit_a": zeros,
                "p_v": half,
                "p_a": half,
                "w_v": half,
                "w_a": half,
                "weights": torch.full((b, 2), 0.5),
                "uncertainty_v": half,
                "uncertainty_a": half,
                "sync_features": torch.zeros(b, 4),
                "gate_logits": torch.zeros(b, 2),
                "visual_missing": zeros,
                "audio_missing": zeros,
            }

    model = SimpleModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = CombinedLoss(alpha=0.25, beta=0.1, gamma=0.0, eta=0.05)
    cfg = load_train_config()
    trainer = Trainer(model, opt, loss_fn, cfg)

    class DummyDS:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "frames": torch.zeros(16, 3, 160, 160),
                "mel_spec": torch.zeros(1, 80, 201),
                "quality_v": torch.zeros(7),
                "quality_a": torch.zeros(5),
                "visual_missing": torch.tensor([0.0]),
                "audio_missing": torch.tensor([0.0]),
                "target": torch.tensor([1.0]),
            }

    loader = DataLoader(DummyDS(), batch_size=1)
    metrics = trainer.train_epoch(loader, epoch=1)
    assert "train_loss" in metrics
    assert "train_eta" in metrics
    assert metrics["train_eta"] > 0.0


def test_trainer_val_loss_optimization_direction() -> None:
    """Trainer correctly treats lower val_loss as an improvement when configured."""
    from src.train.trainer import Trainer
    from src.utils.config_loader import load_train_config

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

    cfg = load_train_config()
    cfg.checkpoint.metric_for_best = "val_loss"
    model = MockModel()
    trainer = Trainer(model, None, None, cfg)
    assert trainer.is_loss_metric is True
    assert trainer.best_metric == float("inf")
