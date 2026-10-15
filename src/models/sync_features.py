"""Computes cross-modal offset cosine similarity c over offsets delta in {-2..2} as a secondary reliability signal, never a standalone label.

Status: implemented — provides SyncFeatureExtractor that computes per-offset
cosine similarities between projected visual and audio embeddings, outputting
[s_0, max_δ s_δ, best_offset, dispersion] as a 4-D feature vector.
Synchronization is a feature, not a verdict (Section 7.3).
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SyncFeatureExtractor(nn.Module):
    """Extracts synchronization features from aligned visual/audio projections.

    For offsets δ ∈ {-2, -1, 0, 1, 2}, computes mean cosine similarity s_δ
    between projected visual and audio embeddings.  Outputs:

    - ``s_0``: cosine similarity at zero offset
    - ``max_δ s_δ``: maximum similarity across offsets
    - ``best_offset``: the offset with highest similarity (normalized)
    - ``dispersion``: standard deviation of similarities across offsets

    **Constraint (Section 7.3):** These features affect the gate and classifier
    but never independently determine the label.  Out-of-sync ≠ fake.

    Parameters
    ----------
    offsets:
        List of temporal offsets to evaluate (default ``[-2, -1, 0, 1, 2]``).
    output_dim:
        Number of output features (default 4).
    """

    def __init__(
        self,
        offsets: list[int] | None = None,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.offsets = offsets or [-2, -1, 0, 1, 2]
        self.output_dim = output_dim

    def _compute_offset_similarity(
        self,
        proj_v: torch.Tensor,
        proj_a: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """Compute mean cosine similarity at a given temporal offset.

        Parameters
        ----------
        proj_v:
            Projected visual embeddings ``(B, T, D)``.
        proj_a:
            Projected audio embeddings ``(B, T, D)``.
        offset:
            Temporal offset δ.  Positive means audio is shifted right.

        Returns
        -------
        torch.Tensor
            Mean cosine similarity per sample ``(B,)``.
        """
        T = proj_v.shape[1]

        if offset >= 0:
            v_start, v_end = 0, T - offset
            a_start, a_end = offset, T
        else:
            v_start, v_end = -offset, T
            a_start, a_end = 0, T + offset

        if v_end <= v_start:
            return torch.zeros(proj_v.shape[0], device=proj_v.device)

        v_slice = proj_v[:, v_start:v_end, :]  # (B, L, D)
        a_slice = proj_a[:, a_start:a_end, :]  # (B, L, D)

        # Normalize for cosine similarity
        v_norm = F.normalize(v_slice, p=2, dim=-1)
        a_norm = F.normalize(a_slice, p=2, dim=-1)

        # Per-bin cosine similarity, then mean over bins
        cos_sim = (v_norm * a_norm).sum(dim=-1)  # (B, L)
        return cos_sim.mean(dim=-1)  # (B,)

    def forward(
        self,
        proj_v: torch.Tensor,
        proj_a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute synchronization features.

        Parameters
        ----------
        proj_v:
            Projected visual embeddings ``(B, T, D)``.
        proj_a:
            Projected audio embeddings ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Sync features ``(B, output_dim)`` = ``[s_0, max_s, best_offset_norm, dispersion]``.
        """
        B = proj_v.shape[0]
        device = proj_v.device

        # Compute similarity at each offset
        similarities = []
        for delta in self.offsets:
            s_delta = self._compute_offset_similarity(proj_v, proj_a, delta)
            similarities.append(s_delta)

        sims = torch.stack(similarities, dim=-1)  # (B, n_offsets)

        # s_0: similarity at zero offset
        zero_idx = self.offsets.index(0) if 0 in self.offsets else len(self.offsets) // 2
        s_0 = sims[:, zero_idx]  # (B,)

        # max_δ s_δ
        max_s, max_idx = sims.max(dim=-1)  # (B,)

        # best_offset: normalized to [-1, 1]
        max_offset = max(abs(o) for o in self.offsets) or 1
        best_offset = torch.tensor(
            [self.offsets[i] for i in max_idx.tolist()],
            device=device,
            dtype=torch.float32,
        ) / max_offset  # (B,)

        # dispersion: std of similarities across offsets
        dispersion = sims.std(dim=-1)  # (B,)

        # Stack into output features
        sync_features = torch.stack(
            [s_0, max_s, best_offset, dispersion], dim=-1
        )  # (B, 4)

        return sync_features
