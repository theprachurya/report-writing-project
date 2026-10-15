"""Staged-training step 2: trains visual-only and audio-only baselines independently.

Status: implemented — provides train_visual_only() and train_audio_only() functions
that train encoder + modality head combinations independently, establishing feasibility
and error profiles before fusion. See AGENTS.md for build order.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config_loader import load_model_config, load_train_config
from src.utils.logging_utils import setup_logging
from src.train.losses import CombinedLoss
from src.train.pipeline import run_training

logger = logging.getLogger(__name__)


class UnimodalVisualModel(nn.Module):
    """Visual-only baseline model (encoder + head).

    Parameters
    ----------
    model_config:
        Model configuration.
    """

    def __init__(self, model_config: Any = None) -> None:
        super().__init__()
        if model_config is None:
            model_config = load_model_config()
        cfg = model_config

        from src.models.visual_encoder import VisualEncoder
        from src.models.modality_heads import ModalityHead

        ve = cfg.visual_encoder
        self.encoder = VisualEncoder(
            backbone=ve.backbone,
            pretrained=ve.pretrained,
            embedding_dim=ve.embedding_dim,
            temporal_bins=ve.temporal_bins,
        )
        self.head = ModalityHead(
            input_dim=ve.embedding_dim,
            hidden_dim=cfg.modality_heads.hidden_dim,
            dropout=cfg.modality_heads.dropout,
        )

    def forward(
        self, frames: torch.Tensor, mel_spec: torch.Tensor | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Forward pass: frames → visual prediction.

        Parameters
        ----------
        frames:
            Face crops ``(B, N, 3, H, W)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Model output with logit, prob, uncertainty.
        """
        z_v = self.encoder(frames)
        out = self.head(z_v)
        batch_size = frames.shape[0]
        return {
            "logit_f": out["logit"],
            "p_f": out["prob"],
            "logit_v": out["logit"],
            "logit_a": out["logit"],  # Dummy for loss compat
            "p_v": out["prob"],
            "p_a": out["prob"],
            "w_v": torch.ones_like(out["prob"]),
            "w_a": torch.zeros_like(out["prob"]),
            "uncertainty_v": out["uncertainty"],
            "uncertainty_a": torch.zeros_like(out["uncertainty"]),
            "sync_features": torch.zeros(frames.shape[0], 4, device=frames.device),
            "gate_logits": torch.zeros(frames.shape[0], 2, device=frames.device),
            "visual_missing": kwargs.get("visual_missing", torch.zeros(batch_size, 1, device=frames.device)),
            "audio_missing": torch.ones(batch_size, 1, device=frames.device),
        }


class UnimodalAudioModel(nn.Module):
    """Audio-only baseline model (encoder + head).

    Parameters
    ----------
    model_config:
        Model configuration.
    """

    def __init__(self, model_config: Any = None) -> None:
        super().__init__()
        if model_config is None:
            model_config = load_model_config()
        cfg = model_config

        from src.models.audio_encoder import AudioEncoder
        from src.models.modality_heads import ModalityHead

        ae = cfg.audio_encoder
        self.encoder = AudioEncoder(
            embedding_dim=ae.embedding_dim,
            temporal_bins=ae.temporal_bins,
            width_multiplier=ae.width_multiplier,
        )
        self.head = ModalityHead(
            input_dim=ae.embedding_dim,
            hidden_dim=cfg.modality_heads.hidden_dim,
            dropout=cfg.modality_heads.dropout,
        )

    def forward(
        self, frames: torch.Tensor | None = None, mel_spec: torch.Tensor | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Forward pass: mel_spec → audio prediction.

        Parameters
        ----------
        mel_spec:
            Log-Mel spectrograms ``(B, 1, n_mels, T_spec)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Model output dict.
        """
        if mel_spec is None:
            raise ValueError("mel_spec is required for the audio-only model")
        z_a = self.encoder(mel_spec)
        out = self.head(z_a)
        batch_size = mel_spec.shape[0]
        return {
            "logit_f": out["logit"],
            "p_f": out["prob"],
            "logit_v": out["logit"],
            "logit_a": out["logit"],
            "p_v": out["prob"],
            "p_a": out["prob"],
            "w_v": torch.zeros_like(out["prob"]),
            "w_a": torch.ones_like(out["prob"]),
            "uncertainty_v": torch.zeros_like(out["uncertainty"]),
            "uncertainty_a": out["uncertainty"],
            "sync_features": torch.zeros(mel_spec.shape[0], 4, device=mel_spec.device),
            "gate_logits": torch.zeros(mel_spec.shape[0], 2, device=mel_spec.device),
            "visual_missing": torch.ones(batch_size, 1, device=mel_spec.device),
            "audio_missing": kwargs.get("audio_missing", torch.zeros(batch_size, 1, device=mel_spec.device)),
        }


def train_unimodal(
    modality: str = "visual",
    config_overrides: dict[str, Any] | None = None,
    *,
    data_config_path: str = "configs/data_config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    train_config_path: str = "configs/train_config.yaml",
    manifest_dir: str = "manifests/splits",
    seed: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train a unimodal baseline model.

    Parameters
    ----------
    modality:
        ``"visual"`` or ``"audio"``.
    config_overrides:
        Optional config overrides.

    Returns
    -------
    dict[str, Any]
        Training history.
    """
    train_config = load_train_config(config_path=train_config_path, overrides=config_overrides)
    model_config = load_model_config(config_path=model_config_path, overrides=config_overrides)

    effective_seed = seed if seed is not None else (train_config.seeds[0] if train_config.seeds else 42)
    logger.info("Training %s-only baseline with seed=%d", modality, effective_seed)

    if modality == "visual":
        model = UnimodalVisualModel(model_config)
    elif modality == "audio":
        model = UnimodalAudioModel(model_config)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Log parameter count
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f M)", n_params, n_params / 1e6)

    result = run_training(
        model=model,
        loss_fn=CombinedLoss(alpha=0.0, beta=0.0, gamma=0.0, eta=0.0),
        stage="unimodal",
        run_name=f"unimodal_{modality}",
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        manifest_dir=manifest_dir,
        seed=effective_seed,
        epochs=epochs,
        config_overrides=config_overrides,
    )
    return {"modality": modality, "n_params": n_params, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unimodal baselines")
    parser.add_argument(
        "--modality", choices=["visual", "audio"], default="visual"
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
    train_unimodal(
        modality=args.modality,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.config or args.train_config,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
        epochs=args.epochs,
    )
