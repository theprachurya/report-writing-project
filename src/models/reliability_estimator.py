"""2-layer MLP producing gate logits -> softmax(w_v, w_a) from [h_v, h_a, c] per Section 7.2. Weights must sum to one.

Status: implemented — provides ReliabilityEstimator that takes pooled feature
statistics, quality features, missingness flags, uncertainty estimates, and
synchronization features, producing softmax-normalized gate weights (w_v, w_a).
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReliabilityEstimator(nn.Module):
    """Reliability gate producing per-window modality weights.

    Per Section 7.2:
        g = MLP([h_v, h_a, c])
        [w_v, w_a] = softmax(g / T_g)
        w_v + w_a = 1

    The input ``[h_v, h_a, c]`` contains:
    - ``h_v``: visual pooled feature statistics + quality features + uncertainty
    - ``h_a``: audio pooled feature statistics + quality features + uncertainty
    - ``c``: synchronization / consistency features

    Parameters
    ----------
    visual_feature_dim:
        Dimension of h_v (pooled stats + quality + uncertainty).
    audio_feature_dim:
        Dimension of h_a (pooled stats + quality + uncertainty).
    sync_feature_dim:
        Dimension of synchronization features c (default 4).
    hidden_dim:
        Hidden layer dimension (default 128).
    dropout:
        Dropout probability (default 0.1).
    gate_temperature:
        Temperature T_g for softmax scaling (default 1.0).
    use_sync_features:
        Whether to include sync features (ablation flag).
    use_quality_features:
        Whether to include quality features (ablation flag).
    quality_visual_dim:
        Dimension of visual quality features (default 7).
    quality_audio_dim:
        Dimension of audio quality features (default 5).
    """

    def __init__(
        self,
        visual_feature_dim: int = 256,
        audio_feature_dim: int = 256,
        sync_feature_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gate_temperature: float = 1.0,
        T_g: float | None = None,
        use_sync_features: bool = True,
        use_quality_features: bool = True,
        quality_visual_dim: int = 7,
        quality_audio_dim: int = 5,
    ) -> None:
        super().__init__()
        self.gate_temperature = T_g if T_g is not None else gate_temperature
        self.T_g = self.gate_temperature
        self.use_sync_features = use_sync_features
        self.use_quality_features = use_quality_features
        self.quality_visual_dim = quality_visual_dim
        self.quality_audio_dim = quality_audio_dim
        self.sync_feature_dim = sync_feature_dim

        # Compute total input dimension
        # h_v: pooled features + uncertainty
        input_dim = visual_feature_dim + 1  # +1 for uncertainty
        # h_a: pooled features + uncertainty
        input_dim += audio_feature_dim + 1
        # Quality features
        if use_quality_features:
            input_dim += quality_visual_dim + quality_audio_dim
        # Sync features
        if use_sync_features:
            input_dim += sync_feature_dim
        # Missingness flags (2: visual_missing, audio_missing)
        input_dim += 2

        self.input_dim = input_dim

        # 2-layer MLP → 2 gate logits
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [g_v, g_a] gate logits
        )

        logger.info(
            "ReliabilityEstimator: input_dim=%d, hidden=%d, T_g=%.2f, "
            "sync=%s, quality=%s",
            input_dim, hidden_dim, self.T_g,
            use_sync_features, use_quality_features,
        )

    def forward(
        self,
        h_v: torch.Tensor,
        h_a: torch.Tensor,
        uncertainty_v: torch.Tensor,
        uncertainty_a: torch.Tensor,
        sync_features: torch.Tensor | None = None,
        quality_v: torch.Tensor | None = None,
        quality_a: torch.Tensor | None = None,
        visual_missing: torch.Tensor | None = None,
        audio_missing: torch.Tensor | None = None,
        T_g: float | None = None,
        gate_temperature: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute reliability gate weights.

        Parameters
        ----------
        h_v:
            Pooled visual features ``(B, D_v)``.
        h_a:
            Pooled audio features ``(B, D_a)``.
        uncertainty_v:
            Visual modality head uncertainty ``(B, 1)``.
        uncertainty_a:
            Audio modality head uncertainty ``(B, 1)``.
        sync_features:
            Synchronization features ``(B, sync_dim)`` or ``None``.
        quality_v:
            Visual quality features ``(B, 7)`` or ``None``.
        quality_a:
            Audio quality features ``(B, 5)`` or ``None``.
        visual_missing:
            Visual modality missing flag ``(B, 1)`` or ``None``.
        audio_missing:
            Audio modality missing flag ``(B, 1)`` or ``None``.
        T_g:
            Optional temperature override for softmax scaling.
        gate_temperature:
            Optional alias for T_g.

        Returns
        -------
        dict[str, torch.Tensor]
            - ``"w_v"``: visual weight ``(B, 1)``
            - ``"w_a"``: audio weight ``(B, 1)``
            - ``"weights"``: ``[w_v, w_a]`` ``(B, 2)``
            - ``"gate_logits"``: raw logits ``(B, 2)``
        """
        B = h_v.shape[0]
        device = h_v.device

        # Resolve temperature T_g
        temp = T_g if T_g is not None else (
            gate_temperature if gate_temperature is not None else self.T_g
        )
        if temp <= 0:
            raise ValueError(f"Temperature T_g must be positive, got {temp}")

        # Missingness flags are always present in the gate input.
        visual_missing_tensor = self._missing_flag(visual_missing, B, device)
        audio_missing_tensor = self._missing_flag(audio_missing, B, device)
        missing_v = visual_missing_tensor[:, 0] >= 0.5
        missing_a = audio_missing_tensor[:, 0] >= 0.5
        both_missing = missing_v & missing_a

        # Cleanly neutralize inputs for missing modalities
        if visual_missing is not None:
            v_mask = missing_v.unsqueeze(-1)
            h_v = h_v.masked_fill(v_mask, 0.0)
            uncertainty_v = uncertainty_v.masked_fill(v_mask, 1.0)
            if quality_v is not None:
                quality_v = quality_v.masked_fill(v_mask, 0.0)

        if audio_missing is not None:
            a_mask = missing_a.unsqueeze(-1)
            h_a = h_a.masked_fill(a_mask, 0.0)
            uncertainty_a = uncertainty_a.masked_fill(a_mask, 1.0)
            if quality_a is not None:
                quality_a = quality_a.masked_fill(a_mask, 0.0)

        if sync_features is not None and (visual_missing is not None or audio_missing is not None):
            either_missing = (missing_v | missing_a).unsqueeze(-1)
            sync_features = sync_features.masked_fill(either_missing, 0.0)

        # Build input: [h_v, uncertainty_v, h_a, uncertainty_a, ...]
        parts = [h_v, uncertainty_v, h_a, uncertainty_a]

        if self.use_quality_features:
            if quality_v is not None:
                parts.append(quality_v)
            else:
                parts.append(torch.zeros(B, self.quality_visual_dim, device=device))
            if quality_a is not None:
                parts.append(quality_a)
            else:
                parts.append(torch.zeros(B, self.quality_audio_dim, device=device))

        if self.use_sync_features:
            if sync_features is not None:
                parts.append(sync_features)
            else:
                parts.append(torch.zeros(B, self.sync_feature_dim, device=device))

        parts.extend([visual_missing_tensor, audio_missing_tensor])

        # Concatenate all inputs
        gate_input = torch.cat(parts, dim=-1)  # (B, input_dim)

        # MLP forward
        gate_logits = self.mlp(gate_input)  # (B, 2)

        # Clean missing modality masking
        # A missing stream must not receive predictive weight. When both streams
        # are missing, use neutral weights; the classifier then sees zero features.
        masked_logits = gate_logits.clone()
        masked_logits[:, 0] = masked_logits[:, 0].masked_fill(missing_v, -1e4)
        masked_logits[:, 1] = masked_logits[:, 1].masked_fill(missing_a, -1e4)
        masked_logits[both_missing] = 0.0

        # Temperature-scaled softmax: softmax(logits / T_g, dim=-1)
        weights = F.softmax(masked_logits / temp, dim=-1)  # (B, 2)
        w_v = weights[:, 0:1]  # (B, 1)
        w_a = weights[:, 1:2]  # (B, 1)

        # Enforce exact bounds on missing modalities
        only_v_missing = (missing_v & ~missing_a).unsqueeze(-1)
        w_v = torch.where(only_v_missing, torch.zeros_like(w_v), w_v)
        w_a = torch.where(only_v_missing, torch.ones_like(w_a), w_a)

        only_a_missing = (missing_a & ~missing_v).unsqueeze(-1)
        w_v = torch.where(only_a_missing, torch.ones_like(w_v), w_v)
        w_a = torch.where(only_a_missing, torch.zeros_like(w_a), w_a)

        both_m = both_missing.unsqueeze(-1)
        w_v = torch.where(both_m, torch.full_like(w_v, 0.5), w_v)
        w_a = torch.where(both_m, torch.full_like(w_a, 0.5), w_a)

        weights = torch.cat([w_v, w_a], dim=-1)

        return {
            "w_v": w_v,
            "w_a": w_a,
            "weights": weights,
            "gate_logits": gate_logits,
            "masked_gate_logits": masked_logits,
        }

    @staticmethod
    def _missing_flag(
        value: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Normalize a missingness flag to a float ``(B, 1)`` tensor."""
        if value is None:
            return torch.zeros(batch_size, 1, device=device)
        return value.to(device=device, dtype=torch.float32).reshape(batch_size, 1)
