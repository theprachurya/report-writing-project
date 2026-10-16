"""Staged-training step 4: trains the full reliability-aware fusion model with modality-specific augmentation.

Status: implemented — provides train_reliability_fusion() for training the full system
with reliability gate, quality features, sync features, and modality-specific augmentation.
Supports encoder freezing for initial epochs. See AGENTS.md for build order.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch

from src.models.full_model import FullModel
from src.train.losses import CombinedLoss
from src.utils.config_loader import load_model_config, load_train_config
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import set_seed
from src.train.pipeline import run_training

logger = logging.getLogger(__name__)


def freeze_encoders(model: FullModel) -> None:
    """Freeze visual and audio encoder parameters.

    Parameters
    ----------
    model:
        The full model whose encoders should be frozen.
    """
    for param in model.visual_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    logger.info("Encoders frozen.")


def unfreeze_encoder_last_blocks(
    model: FullModel, n_blocks: int = 2
) -> None:
    """Unfreeze the last N blocks of each encoder.

    Parameters
    ----------
    model:
        The full model.
    n_blocks:
        Number of trailing blocks to unfreeze.
    """
    # Unfreeze all first, then re-freeze early layers
    for param in model.visual_encoder.parameters():
        param.requires_grad = True
    for param in model.audio_encoder.parameters():
        param.requires_grad = True

    # For MobileNetV3, freeze all but last n_blocks of features
    if hasattr(model.visual_encoder.backbone, "features"):
        features = list(model.visual_encoder.backbone.features)
        for block in features[:-n_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    # For audio encoder, freeze all but last n_blocks of ds_blocks
    if hasattr(model.audio_encoder, "ds_blocks"):
        blocks = list(model.audio_encoder.ds_blocks)
        for block in blocks[:-n_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Encoder last %d blocks unfrozen. Trainable: %d/%d (%.1f%%)",
        n_blocks, n_trainable, n_total, 100 * n_trainable / n_total,
    )


def build_optimizer(
    model: torch.nn.Module,
    train_config: Any,
) -> torch.optim.Optimizer:
    """Build optimizer from training config.

    Parameters
    ----------
    model:
        Model to optimize.
    train_config:
        Training configuration.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.
    """
    opt_cfg = train_config.optimizer
    opt_type = opt_cfg.type.lower() if hasattr(opt_cfg, 'type') else "adamw"
    lr = opt_cfg.lr if hasattr(opt_cfg, 'lr') else 1e-3
    wd = opt_cfg.weight_decay if hasattr(opt_cfg, 'weight_decay') else 1e-4

    # Include currently frozen parameters; the stage runner unfreezes the final
    # encoder blocks after its configured warm-up epochs.
    params = model.parameters()

    if opt_type == "adamw":
        betas = tuple(opt_cfg.betas) if hasattr(opt_cfg, 'betas') else (0.9, 0.999)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    elif opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def train_reliability_fusion(
    config_overrides: dict[str, Any] | None = None,
    *,
    data_config_path: str = "configs/data_config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    train_config_path: str = "configs/train_config.yaml",
    manifest_dir: str = "manifests/splits",
    seed: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train the full reliability-aware fusion model.

    Parameters
    ----------
    config_overrides:
        Optional config overrides.

    Returns
    -------
    dict[str, Any]
        Training results.
    """
    train_config = load_train_config(config_path=train_config_path, overrides=config_overrides)
    model_config = load_model_config(config_path=model_config_path, overrides=config_overrides)

    effective_seed = seed if seed is not None else (train_config.seeds[0] if train_config.seeds else 42)
    logger.info("Training reliability-aware fusion model, seed=%d", effective_seed)

    # Build model
    model = FullModel(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f M)", n_params, n_params / 1e6)

    # Build loss
    loss_cfg = train_config.loss
    # Freeze encoders for initial epochs if configured
    freeze_cfg = train_config.freeze
    freeze_epochs = getattr(freeze_cfg, "freeze_encoders_initial_epochs", 0)
    unfreeze_blocks = getattr(freeze_cfg, "unfreeze_last_n_blocks", 2)
    loss = CombinedLoss(
        alpha=loss_cfg.alpha,
        beta=loss_cfg.beta,
        gamma=loss_cfg.gamma,
        eta=loss_cfg.eta,
        tau_r=loss_cfg.tau_r,
        label_smoothing=getattr(loss_cfg, "label_smoothing", 0.0),
    )
    result = run_training(
        model=model,
        loss_fn=loss,
        stage="reliability_fusion",
        run_name="reliability_fusion",
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        manifest_dir=manifest_dir,
        seed=effective_seed,
        epochs=epochs,
        freeze_encoder_epochs=freeze_epochs,
        unfreeze_last_blocks=unfreeze_blocks,
        config_overrides=config_overrides,
    )

    return {
        "model_params": n_params,
        "loss_weights": {
            "alpha": loss_cfg.alpha,
            "beta": loss_cfg.beta,
            "gamma": loss_cfg.gamma,
            "eta": loss_cfg.eta,
        },
        **result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train reliability-aware fusion model"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--manifest-dir", default="manifests/splits")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    train_reliability_fusion(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.config or args.train_config,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
        epochs=args.epochs,
    )
