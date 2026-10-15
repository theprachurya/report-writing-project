"""Reliability-weighted fusion f = [w_v*z_v_bar, w_a*z_a_bar, c, w_v, w_a] -> MLP -> fused logit p_f.

Status: implemented — provides FusionClassifier supporting multiple fusion modes:
equal_weight, concat, learned_global, and reliability-weighted. Keeps the gate weights
and sync features visible to the classifier. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FusionClassifier(nn.Module):
    """Fused binary classifier with multiple fusion mode support.

    Per Section 7.1:
        f = [w_v * z_v_bar, w_a * z_a_bar, c, w_v, w_a] → 256-D MLP → logit

    Supports ablation baselines (Section 9.1):
    - ``"equal_weight"``: w_v = w_a = 0.5 (baseline 3)
    - ``"concat"``: simple concatenation (baseline 4)
    - ``"learned_global"``: single learned [w_v, w_a] for all samples (baseline 5)
    - ``"reliability"``: per-sample reliability weights (full system, baseline 8)

    Parameters
    ----------
    embedding_dim:
        Dimension of mean-pooled z_v_bar and z_a_bar (default 256).
    sync_feature_dim:
        Dimension of synchronization features c (default 4).
    hidden_dim:
        Hidden layer dimension (default 256).
    dropout:
        Dropout probability (default 0.2).
    fusion_mode:
        One of ``"equal_weight"``, ``"concat"``, ``"learned_global"``,
        ``"reliability"`` (default).
    use_sync_features:
        Whether to include sync features in fusion input.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        sync_feature_dim: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        fusion_mode: str = "reliability",
        use_sync_features: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sync_feature_dim = sync_feature_dim
        self.fusion_mode = fusion_mode
        self.use_sync_features = use_sync_features

        # Compute input dimension based on fusion mode
        if fusion_mode == "concat":
            input_dim = embedding_dim * 2  # [z_v_bar, z_a_bar]
        elif fusion_mode in ("equal_weight", "learned_global", "reliability"):
            input_dim = embedding_dim * 2 + 2  # [w_v*z_v_bar, w_a*z_a_bar, w_v, w_a]
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        if use_sync_features:
            input_dim += sync_feature_dim

        # Learned global weights for baseline 5
        if fusion_mode == "learned_global":
            self.global_gate_logits = nn.Parameter(torch.zeros(2))

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        logger.info(
            "FusionClassifier: mode=%s, input_dim=%d, hidden=%d, sync=%s",
            fusion_mode, input_dim, hidden_dim, use_sync_features,
        )

    def forward(
        self,
        z_v_bar: torch.Tensor,
        z_a_bar: torch.Tensor,
        w_v: torch.Tensor | None = None,
        w_a: torch.Tensor | None = None,
        sync_features: torch.Tensor | None = None,
        visual_missing: torch.Tensor | None = None,
        audio_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse modality features and classify.

        Parameters
        ----------
        z_v_bar:
            Mean-pooled visual features ``(B, embedding_dim)``.
        z_a_bar:
            Mean-pooled audio features ``(B, embedding_dim)``.
        w_v:
            Visual reliability weight ``(B, 1)`` — required for reliability mode.
        w_a:
            Audio reliability weight ``(B, 1)`` — required for reliability mode.
        sync_features:
            Synchronization features ``(B, sync_dim)`` or ``None``.
        visual_missing, audio_missing:
            Optional binary masks ``(B, 1)``. A single missing stream receives
            zero weight; both missing streams receive zero embeddings.

        Returns
        -------
        dict[str, torch.Tensor]
            - ``"logit"``: fused logit ``(B, 1)``
            - ``"prob"``: sigmoid probability ``(B, 1)``
            - ``"w_v"``: visual weight used ``(B, 1)``
            - ``"w_a"``: audio weight used ``(B, 1)``
        """
        B = z_v_bar.shape[0]

        # Determine weights based on fusion mode
        if self.fusion_mode == "equal_weight":
            w_v_used = torch.full((B, 1), 0.5, device=z_v_bar.device)
            w_a_used = torch.full((B, 1), 0.5, device=z_v_bar.device)
        elif self.fusion_mode == "learned_global":
            global_weights = torch.softmax(self.global_gate_logits, dim=0)
            w_v_used = global_weights[0].expand(B, 1)
            w_a_used = global_weights[1].expand(B, 1)
        elif self.fusion_mode == "reliability":
            assert w_v is not None and w_a is not None, (
                "Reliability mode requires w_v and w_a from the reliability estimator"
            )
            w_v_used = w_v
            w_a_used = w_a
        elif self.fusion_mode == "concat":
            w_v_used = torch.ones(B, 1, device=z_v_bar.device)
            w_a_used = torch.ones(B, 1, device=z_v_bar.device)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        if visual_missing is not None or audio_missing is not None:
            visual_missing = (
                torch.zeros(B, 1, device=z_v_bar.device, dtype=torch.bool)
                if visual_missing is None
                else visual_missing.to(device=z_v_bar.device).reshape(B, 1) >= 0.5
            )
            audio_missing = (
                torch.zeros(B, 1, device=z_v_bar.device, dtype=torch.bool)
                if audio_missing is None
                else audio_missing.to(device=z_v_bar.device).reshape(B, 1) >= 0.5
            )
            if self.fusion_mode != "concat":
                w_v_used = w_v_used.masked_fill(visual_missing, 0.0)
                w_a_used = w_a_used.masked_fill(audio_missing, 0.0)
                weight_sum = w_v_used + w_a_used
                has_modality = weight_sum > 0
                w_v_used = torch.where(has_modality, w_v_used / weight_sum.clamp_min(1e-8), w_v_used)
                w_a_used = torch.where(has_modality, w_a_used / weight_sum.clamp_min(1e-8), w_a_used)
            else:
                z_v_bar = z_v_bar.masked_fill(visual_missing, 0.0)
                z_a_bar = z_a_bar.masked_fill(audio_missing, 0.0)

        # Build fusion input
        if self.fusion_mode == "concat":
            parts = [z_v_bar, z_a_bar]
        else:
            # Weighted features + explicit weights
            parts = [
                w_v_used * z_v_bar,
                w_a_used * z_a_bar,
                w_v_used,
                w_a_used,
            ]

        if self.use_sync_features and sync_features is not None:
            parts.append(sync_features)
        elif self.use_sync_features:
            parts.append(torch.zeros(B, self.sync_feature_dim, device=z_v_bar.device))

        fusion_input = torch.cat(parts, dim=-1)

        # Classify
        logit = self.classifier(fusion_input)  # (B, 1)
        prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "prob": prob,
            "w_v": w_v_used,
            "w_a": w_a_used,
        }
