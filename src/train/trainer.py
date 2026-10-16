"""Generic training loop: AMP, gradient accumulation, checkpointing, MLflow/TensorBoard logging, deterministic seeding.

Status: implemented — provides Trainer class with mixed-precision training, gradient
accumulation, best-model checkpointing, early stopping, and experiment logging.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logging_utils import ExperimentLogger, save_config_with_checkpoint
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Generic training loop with AMP, gradient accumulation, and checkpointing.

    Parameters
    ----------
    model:
        The model to train.
    optimizer:
        Optimizer instance.
    scheduler:
        Learning rate scheduler (optional).
    loss_fn:
        Loss function that accepts ``(model_output, target)`` and returns
        a dict with a ``"total"`` key.
    config:
        Training configuration dict or TrainConfig dataclass.
    experiment_logger:
        ExperimentLogger for metric logging (optional).
    device:
        Device to train on (default: auto-detect).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: Any,
        scheduler: Any | None = None,
        experiment_logger: ExperimentLogger | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.exp_logger = experiment_logger

        # Extract config values
        cfg = config if isinstance(config, dict) else vars(config)
        amp_cfg = cfg.get("amp", {})
        if hasattr(amp_cfg, "enabled"):
            self.use_amp = amp_cfg.enabled
        elif isinstance(amp_cfg, dict):
            self.use_amp = amp_cfg.get("enabled", True)
        else:
            self.use_amp = True

        self.grad_accum_steps = cfg.get("gradient_accumulation_steps", 4)

        # Checkpoint config
        ckpt_cfg = cfg.get("checkpoint", {})
        if hasattr(ckpt_cfg, "save_dir"):
            self.ckpt_dir = Path(ckpt_cfg.save_dir)
            self.save_every = ckpt_cfg.save_every_n_epochs
            self.save_best = ckpt_cfg.save_best
            self.metric_for_best = ckpt_cfg.metric_for_best
        elif isinstance(ckpt_cfg, dict):
            self.ckpt_dir = Path(ckpt_cfg.get("save_dir", "artifacts/checkpoints"))
            self.save_every = ckpt_cfg.get("save_every_n_epochs", 5)
            self.save_best = ckpt_cfg.get("save_best", True)
            self.metric_for_best = ckpt_cfg.get("metric_for_best", "val_auroc")
        else:
            self.ckpt_dir = Path("artifacts/checkpoints")
            self.save_every = 5
            self.save_best = True
            self.metric_for_best = "val_auroc"

        # Early stopping
        es_cfg = cfg.get("early_stopping", {})
        if hasattr(es_cfg, "enabled"):
            self.early_stop = es_cfg.enabled
            self.patience = es_cfg.patience
            self.min_delta = es_cfg.min_delta
        elif isinstance(es_cfg, dict):
            self.early_stop = es_cfg.get("enabled", True)
            self.patience = es_cfg.get("patience", 10)
            self.min_delta = es_cfg.get("min_delta", 1e-4)
        else:
            self.early_stop = True
            self.patience = 10
            self.min_delta = 1e-4

        # AMP scaler
        self.use_amp = bool(self.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Tracking
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.freeze_encoder_epochs = 0
        self.unfreeze_last_blocks = 2

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch.

        Parameters
        ----------
        train_loader:
            Training data loader.
        epoch:
            Current epoch number.

        Returns
        -------
        dict[str, float]
            Epoch-level training metrics.
        """
        self.model.train()
        total_loss = 0.0
        loss_components: dict[str, float] = {}
        n_batches = 0

        self.optimizer.zero_grad()

        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._to_device(batch)
            target = batch["target"]

            # Forward pass with AMP
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                model_output = self.model(
                    frames=batch["frames"],
                    mel_spec=batch["mel_spec"],
                    quality_v=batch.get("quality_v"),
                    quality_a=batch.get("quality_a"),
                    visual_missing=batch.get("visual_missing"),
                    audio_missing=batch.get("audio_missing"),
                )
                losses = self.loss_fn(model_output, target)
                group_start = (batch_idx // self.grad_accum_steps) * self.grad_accum_steps
                group_size = min(self.grad_accum_steps, total_batches - group_start)
                loss = losses["total"] / max(group_size, 1)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (
                (batch_idx + 1) % self.grad_accum_steps == 0
                or batch_idx + 1 == total_batches
            ):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += losses["total"].item()
            for k, v in losses.items():
                if k != "total":
                    loss_components[k] = loss_components.get(k, 0.0) + v.item()
            n_batches += 1

        # Average metrics
        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"train_loss": avg_loss}
        for k, v in loss_components.items():
            metrics[f"train_{k}"] = v / max(n_batches, 1)

        if self.exp_logger:
            self.exp_logger.log_metrics(metrics, step=epoch)

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run validation.

        Parameters
        ----------
        val_loader:
            Validation data loader.
        epoch:
            Current epoch number.

        Returns
        -------
        dict[str, float]
            Validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        n_batches = 0

        for batch in val_loader:
            batch = self._to_device(batch)
            target = batch["target"]

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                model_output = self.model(
                    frames=batch["frames"],
                    mel_spec=batch["mel_spec"],
                    quality_v=batch.get("quality_v"),
                    quality_a=batch.get("quality_a"),
                    visual_missing=batch.get("visual_missing"),
                    audio_missing=batch.get("audio_missing"),
                )
                losses = self.loss_fn(model_output, target)

            total_loss += losses["total"].item()
            all_probs.append(model_output["p_f"].cpu())
            all_targets.append(target.cpu())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Compute AUROC if sklearn available
        metrics = {"val_loss": avg_loss}
        try:
            from sklearn.metrics import roc_auc_score
            probs = torch.cat(all_probs).numpy().flatten()
            targets = torch.cat(all_targets).numpy().flatten()
            if len(set(targets)) > 1:
                metrics["val_auroc"] = float(roc_auc_score(targets, probs))
        except ImportError:
            pass

        if self.exp_logger:
            self.exp_logger.log_metrics(metrics, step=epoch)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        config_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Full training loop.

        Parameters
        ----------
        train_loader:
            Training data loader.
        val_loader:
            Validation data loader.
        n_epochs:
            Number of epochs.
        config_dict:
            Config dict to save with checkpoints.

        Returns
        -------
        dict[str, Any]
            Training history.
        """
        history: dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            if epoch == self.freeze_encoder_epochs + 1 and self.freeze_encoder_epochs > 0:
                self._unfreeze_last_encoder_blocks()

            if hasattr(self.loss_fn, "set_epoch"):
                self.loss_fn.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            # Scheduler step: check if ReduceLROnPlateau and pass validation metric
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_metric = val_metrics.get(self.metric_for_best, val_metrics.get("val_loss", 0.0))
                    self.scheduler.step(val_metric)
                else:
                    self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d (%.1fs): train_loss=%.4f, val_loss=%.4f, %s",
                epoch, n_epochs, elapsed,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if k != "val_loss"),
            )

            # Track history
            for k, v in {**train_metrics, **val_metrics}.items():
                history.setdefault(k, []).append(v)

            # Checkpointing
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics, config_dict)

            if self.save_best:
                current_metric = val_metrics.get(self.metric_for_best, -val_metrics["val_loss"])
                if current_metric > self.best_metric + self.min_delta:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, config_dict, is_best=True)
                    logger.info("New best model: %s=%.4f", self.metric_for_best, current_metric)
                else:
                    self.patience_counter += 1

            # Early stopping
            if self.early_stop and self.patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if self.save_best and (self.ckpt_dir / "best_model.pt").is_file():
            logger.info("Restoring best model checkpoint weights from %s", self.ckpt_dir / "best_model.pt")
            try:
                self.load_checkpoint(self.ckpt_dir / "best_model.pt", weights_only=True)
            except Exception as exc:
                logger.warning("Could not restore best model checkpoint: %s", exc)

        if self.exp_logger:
            self.exp_logger.finish()

        return history

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        weights_only: bool = True,
        restore_training_state: bool = False,
    ) -> dict[str, Any]:
        """Load a checkpoint with weights_only support and optional state restoration.

        Parameters
        ----------
        checkpoint_path:
            Path to the checkpoint file.
        weights_only:
            Whether to use PyTorch's weights_only load (default True).
            Falls back safely to False if custom objects are present.
        restore_training_state:
            Whether to restore optimizer, scaler, and scheduler states.

        Returns
        -------
        dict[str, Any]
            The loaded checkpoint dictionary.
        """
        checkpoint_path = Path(checkpoint_path)
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=weights_only)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        if restore_training_state:
            if "optimizer_state_dict" in ckpt and self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None and self.scaler is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt

    def _unfreeze_last_encoder_blocks(self) -> None:
        """Unfreeze the configured final blocks after the encoder warm-up."""
        for name in ("visual_encoder", "audio_encoder"):
            encoder = getattr(self.model, name, None)
            if encoder is None:
                continue
            backbone = getattr(encoder, "backbone", None)
            if backbone is not None and hasattr(backbone, "features"):
                blocks = list(backbone.features)
            elif hasattr(encoder, "ds_blocks"):
                blocks = list(encoder.ds_blocks)
            else:
                blocks = []
            if blocks:
                for block in blocks[-self.unfreeze_last_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
            else:
                for parameter in encoder.parameters():
                    parameter.requires_grad = True
        logger.info(
            "Unfroze the final %d encoder blocks after warm-up",
            self.unfreeze_last_blocks,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        config_dict: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> None:
        """Save a model checkpoint with scaler, scheduler, and RNG state."""
        import random
        import numpy as np

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.ckpt_dir / name

        seed_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "seed_state": seed_state,
            "metrics": metrics,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }, path)

        # Save config alongside checkpoint (reproducibility checklist)
        if config_dict is not None:
            save_config_with_checkpoint(config_dict, self.ckpt_dir)

        logger.info("Checkpoint saved: %s", path)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch dict to the target device, ensuring float32 precision."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                t = v.to(self.device)
                if t.is_floating_point() and t.dtype == torch.float64:
                    t = t.float()
                result[k] = t
            else:
                result[k] = v
        return result
