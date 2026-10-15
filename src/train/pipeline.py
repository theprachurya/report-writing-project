"""Shared executable training path for the staged model baselines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.train.data_pipeline import build_train_val_loaders
from src.data.manifest_dataset import dataset_roots_from_config
from src.train.trainer import Trainer
from src.utils.config_loader import load_yaml
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def build_optimizer(model: torch.nn.Module, train_config: Any) -> torch.optim.Optimizer:
    """Create the configured optimizer, including initially frozen parameters."""
    opt = train_config.optimizer
    common = {"lr": opt.lr, "weight_decay": opt.weight_decay}
    name = opt.type.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), betas=tuple(opt.betas), eps=opt.eps, **common)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), betas=tuple(opt.betas), eps=opt.eps, **common)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **common)
    raise ValueError(f"Unknown optimizer type in train config: {opt.type}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_config: Any) -> Any | None:
    """Build the configured learning-rate scheduler."""
    cfg = train_config.scheduler
    name = cfg.type.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.T_0), eta_min=cfg.eta_min
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.T_0), gamma=0.1)
    if name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, cfg.T_0), T_mult=max(1, cfg.T_mult), eta_min=cfg.eta_min
        )
    if name in {"none", "disabled"}:
        return None
    raise ValueError(f"Unknown scheduler type in train config: {cfg.type}")


def run_training(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    stage: str,
    run_name: str,
    data_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    manifest_dir: str | Path,
    seed: int | None = None,
    epochs: int | None = None,
    freeze_encoder_epochs: int = 0,
    unfreeze_last_blocks: int = 2,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train from committed train/validation manifests and save a best checkpoint."""
    from src.utils.config_loader import load_data_config, load_train_config

    data_config = load_data_config(config_path=data_config_path)
    train_config = load_train_config(config_path=train_config_path, overrides=config_overrides)
    seed = int(seed if seed is not None else (train_config.seeds[0] if train_config.seeds else 42))
    set_seed(seed)
    logger.info("Starting %s run with seed=%d", run_name, seed)

    roots = set(dataset_roots_from_config(data_config))
    if stage != "unimodal" or run_name == "unimodal_audio":
        allowed_datasets = {"fakeavceleb"}
    else:
        allowed_datasets = {"fakeavceleb"} if "fakeavceleb" in roots else {"dfdc"}
    train_loader, val_loader = build_train_val_loaders(
        data_config,
        train_config,
        manifest_dir=manifest_dir,
        seed=seed,
        allowed_datasets=allowed_datasets,
    )

    if freeze_encoder_epochs > 0 and hasattr(model, "visual_encoder"):
        for parameter in model.visual_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.audio_encoder.parameters():
            parameter.requires_grad = False

    base_ckpt_dir = Path(train_config.checkpoint.save_dir)
    run_ckpt_dir = base_ckpt_dir / run_name
    train_config.checkpoint.save_dir = str(run_ckpt_dir)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    trainer = Trainer(model, optimizer, loss_fn, train_config, scheduler=scheduler)
    trainer.freeze_encoder_epochs = int(freeze_encoder_epochs)
    trainer.unfreeze_last_blocks = int(unfreeze_last_blocks)

    raw_train_config = load_yaml(train_config_path)
    if config_overrides:
        from src.utils.config_loader import merge_configs
        raw_train_config = merge_configs(raw_train_config, config_overrides)
    raw_config = {
        "data": load_yaml(data_config_path),
        "model": load_yaml(model_config_path),
        "train": raw_train_config,
        "run": {"name": run_name, "stage": stage, "seed": seed, "manifest_dir": str(manifest_dir)},
    }
    raw_config["train"].setdefault("checkpoint", {})["save_dir"] = str(run_ckpt_dir)
    epoch_count = int(epochs if epochs is not None else train_config.epochs.get(stage, train_config.epochs.get("unimodal", 1)))
    history = trainer.fit(train_loader, val_loader, epoch_count, config_dict=raw_config)

    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_ckpt_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "run_name": run_name,
        "seed": seed,
        "epochs_completed": len(history.get("train_loss", [])),
        "checkpoint": str(run_ckpt_dir / "best_model.pt"),
        "history": history,
    }
