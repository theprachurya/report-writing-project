"""Staged-training step 3: trains concatenation / equal-weight fusion baseline before the gate is introduced.

Status: implemented — provides FixedFusionModel and train_fixed_fusion() for training
the fixed-weight fusion baseline. This must exist and be measured before the reliability
gate is introduced (Section 8). See AGENTS.md for build order.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch
import torch.nn as nn

from src.models.alignment_projector import AlignmentProjector
from src.models.audio_encoder import AudioEncoder
from src.models.fusion_classifier import FusionClassifier
from src.models.modality_heads import ModalityHeads
from src.models.sync_features import SyncFeatureExtractor
from src.models.visual_encoder import VisualEncoder
from src.utils.config_loader import load_model_config, load_train_config
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import set_seed
from src.train.losses import CombinedLoss
from src.train.pipeline import run_training

logger = logging.getLogger(__name__)


class FixedFusionModel(nn.Module):
    """Fixed-weight fusion model (baselines 3-5).

    Uses equal weights or learned global weights instead of per-sample
    reliability estimation.

    Parameters
    ----------
    model_config:
        Model configuration.
    fusion_mode:
        ``"equal_weight"``, ``"concat"``, or ``"learned_global"``.
    """

    def __init__(
        self,
        model_config: Any = None,
        fusion_mode: str = "equal_weight",
    ) -> None:
        super().__init__()
        if model_config is None:
            model_config = load_model_config()
        cfg = model_config

        ve = cfg.visual_encoder
        ae = cfg.audio_encoder
        ap = cfg.alignment_projector
        sf = cfg.sync_features
        mh = cfg.modality_heads
        fc = cfg.fusion_classifier

        ab = cfg.ablation if hasattr(cfg, "ablation") else None
        use_sync = ab.use_sync_features if (ab and hasattr(ab, "use_sync_features")) else True
        self.use_sync = use_sync
        self.sync_feature_dim = sf.output_dim

        self.visual_encoder = VisualEncoder(
            backbone=ve.backbone,
            pretrained=ve.pretrained,
            embedding_dim=ve.embedding_dim,
            temporal_bins=ve.temporal_bins,
        )
        self.audio_encoder = AudioEncoder(
            embedding_dim=ae.embedding_dim,
            temporal_bins=ae.temporal_bins,
            width_multiplier=ae.width_multiplier,
        )
        self.alignment_projector = AlignmentProjector(
            input_dim=ap.input_dim,
            projection_dim=ap.projection_dim,
        )
        self.sync_extractor = SyncFeatureExtractor(
            offsets=sf.offsets if hasattr(sf, 'offsets') else [-2, -1, 0, 1, 2],
            output_dim=sf.output_dim,
        )
        self.modality_heads = ModalityHeads(
            input_dim=mh.input_dim,
            hidden_dim=mh.hidden_dim,
            dropout=mh.dropout,
        )
        self.fusion_classifier = FusionClassifier(
            embedding_dim=ve.embedding_dim,
            sync_feature_dim=sf.output_dim,
            hidden_dim=fc.hidden_dim,
            dropout=fc.dropout,
            fusion_mode=fusion_mode,
            use_sync_features=use_sync,
        )

    def forward(
        self,
        frames: torch.Tensor,
        mel_spec: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the fixed-fusion model.

        Parameters
        ----------
        frames:
            Face crops ``(B, N, 3, H, W)``.
        mel_spec:
            Log-Mel spectrograms ``(B, 1, n_mels, T_spec)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Model output dict compatible with CombinedLoss.
        """
        z_v = self.visual_encoder(frames)
        z_a = self.audio_encoder(mel_spec)

        visual_missing = kwargs.get("visual_missing")
        audio_missing = kwargs.get("audio_missing")
        if visual_missing is not None:
            z_v = z_v.masked_fill(visual_missing.reshape(-1, 1, 1).to(z_v.device) >= 0.5, 0.0)
        if audio_missing is not None:
            z_a = z_a.masked_fill(audio_missing.reshape(-1, 1, 1).to(z_a.device) >= 0.5, 0.0)

        proj_v, proj_a = self.alignment_projector(z_v, z_a)
        sync_features = self.sync_extractor(proj_v, proj_a)
        unavailable = torch.zeros(frames.shape[0], 1, dtype=torch.bool, device=frames.device)
        if visual_missing is not None:
            unavailable |= visual_missing.reshape(-1, 1).to(frames.device) >= 0.5
        if audio_missing is not None:
            unavailable |= audio_missing.reshape(-1, 1).to(frames.device) >= 0.5
        sync_features = sync_features.masked_fill(unavailable, 0.0)

        head_out = self.modality_heads(
            z_v, z_a,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

        z_v_bar = z_v.mean(dim=1)
        z_a_bar = z_a.mean(dim=1)
        if visual_missing is not None:
            z_v_bar = z_v_bar.masked_fill(visual_missing.reshape(-1, 1).to(frames.device) >= 0.5, 0.0)
        if audio_missing is not None:
            z_a_bar = z_a_bar.masked_fill(audio_missing.reshape(-1, 1).to(frames.device) >= 0.5, 0.0)

        fusion_out = self.fusion_classifier(
            z_v_bar=z_v_bar,
            z_a_bar=z_a_bar,
            sync_features=sync_features if self.use_sync else None,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

        return {
            "p_f": fusion_out["prob"],
            "logit_f": fusion_out["logit"],
            "p_v": head_out["visual"]["prob"],
            "p_a": head_out["audio"]["prob"],
            "logit_v": head_out["visual"]["logit"],
            "logit_a": head_out["audio"]["logit"],
            "w_v": fusion_out["w_v"],
            "w_a": fusion_out["w_a"],
            "uncertainty_v": head_out["visual"]["uncertainty"],
            "uncertainty_a": head_out["audio"]["uncertainty"],
            "sync_features": sync_features,
            "gate_logits": torch.zeros(frames.shape[0], 2, device=frames.device),
            "visual_missing": kwargs.get("visual_missing", torch.zeros(frames.shape[0], 1, device=frames.device)),
            "audio_missing": kwargs.get("audio_missing", torch.zeros(frames.shape[0], 1, device=frames.device)),
        }


def train_fixed_fusion(
    fusion_mode: str = "equal_weight",
    config_overrides: dict[str, Any] | None = None,
    *,
    data_config_path: str = "configs/data_config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    train_config_path: str = "configs/train_config.yaml",
    manifest_dir: str = "manifests/splits",
    seed: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train a fixed-fusion baseline model.

    Parameters
    ----------
    fusion_mode:
        ``"equal_weight"``, ``"concat"``, or ``"learned_global"``.
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
    logger.info("Training fixed fusion baseline: mode=%s, seed=%d", fusion_mode, effective_seed)

    model = FixedFusionModel(model_config, fusion_mode=fusion_mode)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f M)", n_params, n_params / 1e6)

    loss_cfg = train_config.loss
    loss_fn = CombinedLoss(
        alpha=loss_cfg.alpha,
        beta=0.0,
        gamma=0.0,
        eta=0.0,
        label_smoothing=getattr(loss_cfg, "label_smoothing", 0.0),
    )
    result = run_training(
        model=model,
        loss_fn=loss_fn,
        stage="fixed_fusion",
        run_name=f"fixed_fusion_{fusion_mode}",
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        manifest_dir=manifest_dir,
        seed=effective_seed,
        epochs=epochs,
        config_overrides=config_overrides,
    )
    return {"fusion_mode": fusion_mode, "n_params": n_params, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fixed fusion baseline")
    parser.add_argument(
        "--mode", choices=["equal_weight", "concat", "learned_global", "learned_fixed"],
        default="equal_weight",
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
    train_fixed_fusion(
        fusion_mode=args.mode,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.config or args.train_config,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
        epochs=args.epochs,
    )
