"""Small linear classifiers on pooled z_v/z_a producing unimodal probabilities p_v, p_a and per-window uncertainty.

Status: implemented — provides ModalityHead for single-modality classification and
ModalityHeads container for both visual and audio heads. Each head outputs a logit,
probability, and uncertainty estimate from mean-pooled encoder features.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModalityHead(nn.Module):
    """Single-modality binary classification head.

    Takes mean-pooled encoder features and produces a logit, probability,
    and uncertainty estimate.

    Parameters
    ----------
    input_dim:
        Dimension of the mean-pooled encoder output (default 256).
    hidden_dim:
        Hidden layer dimension (default 128).
    dropout:
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary logit
        )

    def forward(
        self, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the modality head.

        Parameters
        ----------
        z:
            Encoder output of shape ``(B, T, D)`` or ``(B, D)`` (already pooled).

        Returns
        -------
        dict[str, torch.Tensor]
            - ``"logit"``: raw logit ``(B, 1)``
            - ``"prob"``: sigmoid probability ``(B, 1)``
            - ``"uncertainty"``: 1 - |2p - 1| confidence proxy ``(B, 1)``
            - ``"pooled_features"``: mean-pooled features ``(B, D)``
        """
        # Mean pool over temporal bins if needed
        if z.dim() == 3:
            pooled = z.mean(dim=1)  # (B, D)
        else:
            pooled = z  # (B, D)

        logit = self.classifier(pooled)  # (B, 1)
        prob = torch.sigmoid(logit)

        # Uncertainty: high when prob ≈ 0.5, low when confident
        # Uses 1 - |2p - 1| so uncertainty ∈ [0, 1]
        uncertainty = 1.0 - torch.abs(2.0 * prob - 1.0)

        return {
            "logit": logit,
            "prob": prob,
            "uncertainty": uncertainty,
            "pooled_features": pooled,
        }


class ModalityHeads(nn.Module):
    """Container for visual and audio modality heads.

    Parameters
    ----------
    input_dim:
        Encoder output dimension (default 256).
    hidden_dim:
        Hidden layer dimension (default 128).
    dropout:
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_head = ModalityHead(input_dim, hidden_dim, dropout)
        self.audio_head = ModalityHead(input_dim, hidden_dim, dropout)

    def forward(
        self,
        z_v: torch.Tensor,
        z_a: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass through both modality heads.

        Parameters
        ----------
        z_v:
            Visual encoder output ``(B, T, D)`` or ``(B, D)``.
        z_a:
            Audio encoder output ``(B, T, D)`` or ``(B, D)``.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            ``{"visual": {...}, "audio": {...}}`` where each sub-dict contains
            ``logit``, ``prob``, ``uncertainty``, ``pooled_features``.
        """
        return {
            "visual": self.visual_head(z_v),
            "audio": self.audio_head(z_a),
        }
