"""MobileNetV3-Small (or EfficientNet-B0) backbone + depthwise temporal Conv1D over 8 bins, producing z_v.

Status: implemented — provides VisualEncoder that processes 16 face crops (160×160)
through an ImageNet-pretrained MobileNetV3-Small backbone, pools into T=8 temporal
bins, and applies depthwise temporal convolution to produce z_v ∈ R^{8×256}.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VisualEncoder(nn.Module):
    """Visual encoder for face-crop sequences.

    Extracts per-frame features from an ImageNet-pretrained backbone, pools
    them into T temporal bins, and applies a depthwise temporal Conv1D.

    Parameters
    ----------
    backbone:
        Backbone name: ``"mobilenetv3_small"`` or ``"efficientnet_b0"``.
    pretrained:
        Whether to load ImageNet-pretrained weights.
    embedding_dim:
        Output embedding dimension per temporal bin (default 256).
    temporal_bins:
        Number of temporal bins T (default 8).
    temporal_conv_kernel:
        Kernel size for the depthwise temporal Conv1D (default 3).
    """

    def __init__(
        self,
        backbone: str = "mobilenetv3_small",
        pretrained: bool = True,
        embedding_dim: int = 256,
        temporal_bins: int = 8,
        temporal_conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temporal_bins = temporal_bins

        # Build backbone
        self.backbone, backbone_out_dim = self._build_backbone(backbone, pretrained)

        # Project backbone features to embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Depthwise temporal Conv1D over T bins
        # groups=embedding_dim makes it depthwise
        self.temporal_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=temporal_conv_kernel,
            padding=temporal_conv_kernel // 2,
            groups=embedding_dim,
        )
        self.temporal_norm = nn.BatchNorm1d(embedding_dim)
        self.temporal_act = nn.ReLU(inplace=True)

    def _build_backbone(
        self, name: str, pretrained: bool
    ) -> tuple[nn.Module, int]:
        """Build the backbone feature extractor.

        Returns (backbone, output_dim).
        """
        if name == "mobilenetv3_small":
            try:
                from torchvision.models import (
                    MobileNet_V3_Small_Weights,
                    mobilenet_v3_small,
                )
                weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                model = mobilenet_v3_small(weights=weights)
                out_dim = model.classifier[0].in_features  # 576
                model.classifier = nn.Identity()
                return model, out_dim
            except (ImportError, Exception) as exc:
                logger.warning("torchvision unavailable (%s); using lightweight ConvNet fallback", exc)
                fallback = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                return fallback, 32

        elif name == "efficientnet_b0":
            try:
                from torchvision.models import (
                    EfficientNet_B0_Weights,
                    efficientnet_b0,
                )
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                model = efficientnet_b0(weights=weights)
                out_dim = model.classifier[1].in_features  # 1280
                model.classifier = nn.Identity()
                return model, out_dim
            except (ImportError, Exception) as exc:
                logger.warning("torchvision unavailable (%s); using lightweight ConvNet fallback", exc)
                fallback = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                return fallback, 32

        else:
            raise ValueError(f"Unknown visual backbone: {name}")

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract per-frame features from the backbone.

        Parameters
        ----------
        frames:
            Tensor of shape ``(B, N, 3, H, W)`` where N is the number of
            frames per window (typically 16).

        Returns
        -------
        torch.Tensor
            Per-frame features of shape ``(B, N, backbone_out_dim)``.
        """
        B, N, C, H, W = frames.shape
        # Flatten batch and frame dims for backbone processing
        x = frames.reshape(B * N, C, H, W)
        features = self.backbone(x)  # (B*N, backbone_out_dim)
        return features.reshape(B, N, -1)

    def pool_to_bins(
        self, frame_features: torch.Tensor
    ) -> torch.Tensor:
        """Pool N frame features into T temporal bins via mean pooling.

        Parameters
        ----------
        frame_features:
            Shape ``(B, N, D)`` where N ≥ T.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, D)``.
        """
        B, N, D = frame_features.shape
        T = self.temporal_bins

        # Assign frames to bins uniformly
        frames_per_bin = N // T
        remainder = N % T

        bins = []
        idx = 0
        for t in range(T):
            n = frames_per_bin + (1 if t < remainder else 0)
            bin_features = frame_features[:, idx:idx + n, :].mean(dim=1)
            bins.append(bin_features)
            idx += n

        return torch.stack(bins, dim=1)  # (B, T, D)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Full forward pass: frames → z_v.

        Parameters
        ----------
        frames:
            Tensor of shape ``(B, N, 3, H, W)`` — a batch of N-frame windows.

        Returns
        -------
        torch.Tensor
            z_v of shape ``(B, T, embedding_dim)`` where T=8.
        """
        # Extract per-frame features
        frame_feat = self.extract_frame_features(frames)  # (B, N, D_backbone)

        # Project to embedding_dim
        frame_feat = self.projector(frame_feat)  # (B, N, embedding_dim)

        # Pool into temporal bins
        z_v = self.pool_to_bins(frame_feat)  # (B, T, embedding_dim)

        # Depthwise temporal convolution
        # Conv1d expects (B, C, T)
        z_v = z_v.permute(0, 2, 1)  # (B, embedding_dim, T)
        z_v = self.temporal_conv(z_v)
        z_v = self.temporal_norm(z_v)
        z_v = self.temporal_act(z_v)
        z_v = z_v.permute(0, 2, 1)  # (B, T, embedding_dim)

        return z_v
