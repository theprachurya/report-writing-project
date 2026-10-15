"""Separate linear layers projecting z_v and z_a into a shared 128-D comparison space.

Status: implemented — provides AlignmentProjector with separate learned projections
for visual and audio embeddings into a shared space for synchronization feature
computation. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AlignmentProjector(nn.Module):
    """Projects visual and audio embeddings into a shared comparison space.

    Per Section 7.1: separate linear layers map z_v and z_a to 128-D shared
    space.  This enables cross-modal comparison without pretending raw
    modalities share a coordinate system.

    Parameters
    ----------
    input_dim:
        Input embedding dimension (must match encoder output, default 256).
    projection_dim:
        Shared projection space dimension (default 128).
    """

    def __init__(
        self,
        input_dim: int = 256,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim

        # Separate projectors for each modality
        self.visual_projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def project_visual(self, z_v: torch.Tensor) -> torch.Tensor:
        """Project visual embeddings to the shared space.

        Parameters
        ----------
        z_v:
            Visual embeddings of shape ``(B, T, input_dim)``.

        Returns
        -------
        torch.Tensor
            Projected visual embeddings of shape ``(B, T, projection_dim)``.
        """
        return self.visual_projector(z_v)

    def project_audio(self, z_a: torch.Tensor) -> torch.Tensor:
        """Project audio embeddings to the shared space.

        Parameters
        ----------
        z_a:
            Audio embeddings of shape ``(B, T, input_dim)``.

        Returns
        -------
        torch.Tensor
            Projected audio embeddings of shape ``(B, T, projection_dim)``.
        """
        return self.audio_projector(z_a)

    def forward(
        self,
        z_v: torch.Tensor,
        z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both modalities into the shared space.

        Parameters
        ----------
        z_v:
            Visual embeddings ``(B, T, input_dim)``.
        z_a:
            Audio embeddings ``(B, T, input_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(proj_v, proj_a)`` each of shape ``(B, T, projection_dim)``.
        """
        proj_v = self.project_visual(z_v)
        proj_a = self.project_audio(z_a)
        return proj_v, proj_a
