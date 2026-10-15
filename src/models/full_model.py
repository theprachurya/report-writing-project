"""Wires encoders, projector, sync features, modality heads, reliability estimator, and fusion classifier into the end-to-end system (baseline #8).

Status: implemented — provides FullModel that orchestrates the complete pipeline from
raw face crops and log-Mel spectrograms through to fused predictions, with configurable
ablation flags and missing-modality support. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from src.models.alignment_projector import AlignmentProjector
from src.models.audio_encoder import AudioEncoder
from src.models.fusion_classifier import FusionClassifier
from src.models.modality_heads import ModalityHeads
from src.models.reliability_estimator import ReliabilityEstimator
from src.models.sync_features import SyncFeatureExtractor
from src.models.visual_encoder import VisualEncoder

logger = logging.getLogger(__name__)


class FullModel(nn.Module):
    """End-to-end reliability-aware audio-visual deepfake detector.

    Wires together all sub-modules from Section 7 into a single forward pass:

    1. Visual encoder → z_v
    2. Audio encoder → z_a
    3. Alignment projector → projected embeddings
    4. Sync feature extractor → c
    5. Modality heads → p_v, p_a, uncertainties
    6. Reliability estimator → w_v, w_a
    7. Fusion classifier → p_f

    Supports missing-modality masking, all fusion modes, and ablation flags.

    Parameters
    ----------
    model_config:
        Model configuration object or dict with architecture parameters.
    """

    def __init__(self, model_config: Any = None) -> None:
        super().__init__()

        # Extract config values (support both dataclass and dict)
        if model_config is None:
            from src.utils.config_loader import load_model_config
            model_config = load_model_config()

        cfg = model_config
        if isinstance(cfg, dict):
            # Convert dict access
            class _D:
                def __init__(self, d):
                    self.__dict__.update(d)
                    for k, v in d.items():
                        if isinstance(v, dict):
                            setattr(self, k, _D(v))
            cfg = _D(cfg)

        ve = cfg.visual_encoder
        ae = cfg.audio_encoder
        ap = cfg.alignment_projector
        sf = cfg.sync_features
        mh = cfg.modality_heads
        qf = cfg.quality_features
        re = cfg.reliability_estimator
        fc = cfg.fusion_classifier
        ab = cfg.ablation

        # Sub-modules
        self.visual_encoder = VisualEncoder(
            backbone=ve.backbone,
            pretrained=ve.pretrained,
            embedding_dim=ve.embedding_dim,
            temporal_bins=ve.temporal_bins,
            temporal_conv_kernel=ve.temporal_conv.get("kernel_size", 3)
            if isinstance(ve.temporal_conv, dict)
            else getattr(ve.temporal_conv, "kernel_size", 3),
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

        use_sync = ab.use_sync_features if hasattr(ab, 'use_sync_features') else True
        use_quality = ab.use_quality_features if hasattr(ab, 'use_quality_features') else True
        fusion_mode = ab.fusion_mode if hasattr(ab, 'fusion_mode') else "reliability"
        use_gate = ab.use_reliability_gate if hasattr(ab, 'use_reliability_gate') else True
        if not use_gate and fusion_mode == "reliability":
            fusion_mode = "equal_weight"

        self.reliability_estimator = ReliabilityEstimator(
            visual_feature_dim=ve.embedding_dim,
            audio_feature_dim=ae.embedding_dim,
            sync_feature_dim=sf.output_dim,
            hidden_dim=re.hidden_dim,
            dropout=re.dropout,
            gate_temperature=re.gate_temperature,
            use_sync_features=use_sync,
            use_quality_features=use_quality,
            quality_visual_dim=qf.visual_dim,
            quality_audio_dim=qf.audio_dim,
        )

        self.fusion_classifier = FusionClassifier(
            embedding_dim=ve.embedding_dim,
            sync_feature_dim=sf.output_dim,
            hidden_dim=fc.hidden_dim,
            dropout=fc.dropout,
            fusion_mode=fusion_mode,
            use_sync_features=use_sync,
        )

        self.fusion_mode = fusion_mode
        self.use_sync = use_sync
        self.use_quality = use_quality

    def forward(
        self,
        frames: torch.Tensor,
        mel_spec: torch.Tensor,
        quality_v: torch.Tensor | None = None,
        quality_a: torch.Tensor | None = None,
        visual_missing: torch.Tensor | None = None,
        audio_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        frames:
            Face crops ``(B, N, 3, H, W)`` where N=16 frames.
        mel_spec:
            Log-Mel spectrograms ``(B, 1, n_mels, T_spec)``.
        quality_v:
            Visual quality features ``(B, 7)`` or ``None``.
        quality_a:
            Audio quality features ``(B, 5)`` or ``None``.
        visual_missing:
            Visual missing flag ``(B, 1)`` or ``None``.
        audio_missing:
            Audio missing flag ``(B, 1)`` or ``None``.

        Returns
        -------
        dict[str, torch.Tensor]
            - ``"p_f"``: fused probability ``(B, 1)``
            - ``"logit_f"``: fused logit ``(B, 1)``
            - ``"p_v"``: visual probability ``(B, 1)``
            - ``"p_a"``: audio probability ``(B, 1)``
            - ``"logit_v"``: visual logit ``(B, 1)``
            - ``"logit_a"``: audio logit ``(B, 1)``
            - ``"w_v"``: visual weight ``(B, 1)``
            - ``"w_a"``: audio weight ``(B, 1)``
            - ``"uncertainty_v"``: visual uncertainty ``(B, 1)``
            - ``"uncertainty_a"``: audio uncertainty ``(B, 1)``
            - ``"sync_features"``: sync features ``(B, 4)``
            - ``"gate_logits"``: raw gate logits ``(B, 2)``
        """
        B = frames.shape[0]
        device = frames.device

        # Apply missing-modality masking
        if visual_missing is not None:
            vm = visual_missing.view(B, 1, 1, 1, 1)
            frames = frames * (1.0 - vm.float())
        if audio_missing is not None:
            am = audio_missing.view(B, 1, 1, 1) if mel_spec.dim() == 4 else audio_missing.view(B, 1, 1)
            mel_spec = mel_spec * (1.0 - am.float())

        # 1. Encode
        z_v = self.visual_encoder(frames)           # (B, T, D)
        z_a = self.audio_encoder(mel_spec)           # (B, T, D)

        # Remove encoder bias/normalization outputs from unavailable streams.
        if visual_missing is not None:
            z_v = z_v.masked_fill(visual_missing.reshape(B, 1, 1).to(z_v.device) >= 0.5, 0.0)
        if audio_missing is not None:
            z_a = z_a.masked_fill(audio_missing.reshape(B, 1, 1).to(z_a.device) >= 0.5, 0.0)

        # 2. Project to shared space
        proj_v, proj_a = self.alignment_projector(z_v, z_a)  # (B, T, D_proj)

        # 3. Sync features
        sync_features = self.sync_extractor(proj_v, proj_a)  # (B, 4)
        unavailable = torch.zeros(B, 1, dtype=torch.bool, device=device)
        if visual_missing is not None:
            unavailable |= visual_missing.reshape(B, 1).to(device) >= 0.5
        if audio_missing is not None:
            unavailable |= audio_missing.reshape(B, 1).to(device) >= 0.5
        sync_features = sync_features.masked_fill(unavailable, 0.0)

        # 4. Modality heads
        head_outputs = self.modality_heads(z_v, z_a)
        p_v = head_outputs["visual"]["prob"]
        p_a = head_outputs["audio"]["prob"]
        logit_v = head_outputs["visual"]["logit"]
        logit_a = head_outputs["audio"]["logit"]
        uncertainty_v = head_outputs["visual"]["uncertainty"]
        uncertainty_a = head_outputs["audio"]["uncertainty"]
        pooled_v = head_outputs["visual"]["pooled_features"]
        pooled_a = head_outputs["audio"]["pooled_features"]

        # 5. Reliability gate
        gate_out = self.reliability_estimator(
            h_v=pooled_v,
            h_a=pooled_a,
            uncertainty_v=uncertainty_v,
            uncertainty_a=uncertainty_a,
            sync_features=sync_features if self.use_sync else None,
            quality_v=quality_v if self.use_quality else None,
            quality_a=quality_a if self.use_quality else None,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )
        w_v = gate_out["w_v"]
        w_a = gate_out["w_a"]

        # 6. Mean-pool encoder outputs for fusion
        z_v_bar = z_v.mean(dim=1)  # (B, D)
        z_a_bar = z_a.mean(dim=1)  # (B, D)

        # 7. Fusion classifier
        fusion_out = self.fusion_classifier(
            z_v_bar=z_v_bar,
            z_a_bar=z_a_bar,
            w_v=w_v,
            w_a=w_a,
            sync_features=sync_features if self.use_sync else None,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

        return {
            "p_f": fusion_out["prob"],
            "logit_f": fusion_out["logit"],
            "p_v": p_v,
            "p_a": p_a,
            "logit_v": logit_v,
            "logit_a": logit_a,
            "w_v": fusion_out["w_v"],
            "w_a": fusion_out["w_a"],
            "uncertainty_v": uncertainty_v,
            "uncertainty_a": uncertainty_a,
            "sync_features": sync_features,
            "gate_logits": gate_out["gate_logits"],
            "visual_missing": (
                visual_missing.reshape(B, 1).to(device=device, dtype=torch.float32)
                if visual_missing is not None else torch.zeros(B, 1, device=device)
            ),
            "audio_missing": (
                audio_missing.reshape(B, 1).to(device=device, dtype=torch.float32)
                if audio_missing is not None else torch.zeros(B, 1, device=device)
            ),
        }
