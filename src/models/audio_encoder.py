"""Depthwise-separable 2D CNN over log-Mel patches, producing z_a.

Status: implemented — provides AudioEncoder that processes 80-bin log-Mel
spectrograms through a depthwise-separable 2D CNN, pools into T=8 temporal
bins, producing z_a ∈ R^{8×256}. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2D convolution block.

    Consists of a depthwise convolution followed by a pointwise (1×1)
    convolution, with batch normalization and ReLU after each.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    kernel_size:
        Kernel size for the depthwise convolution.
    stride:
        Stride for the depthwise convolution.
    padding:
        Padding for the depthwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depthwise-separable conv.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x


class AudioEncoder(nn.Module):
    """Audio encoder using depthwise-separable 2D CNN on log-Mel spectrograms.

    Processes ``(B, 1, n_mels, T_spec)`` log-Mel spectrograms through a stack
    of DS-Conv blocks, pools into T=8 temporal bins, producing z_a ∈ R^{8×256}.

    Parameters
    ----------
    n_mels:
        Number of Mel frequency bins (default 80).
    embedding_dim:
        Output embedding dimension per temporal bin (default 256).
    temporal_bins:
        Number of temporal bins T (default 8).
    width_multiplier:
        Channel width multiplier (default 1.0).
    """

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 256,
        temporal_bins: int = 8,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temporal_bins = temporal_bins

        # Channel widths scaled by width_multiplier
        def _ch(base: int) -> int:
            return max(8, int(base * width_multiplier))

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, _ch(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(_ch(32)),
            nn.ReLU(inplace=True),
        )

        # Stack of depthwise-separable conv blocks with progressive downsampling
        self.ds_blocks = nn.Sequential(
            DepthwiseSeparableConv2d(_ch(32), _ch(64), stride=2),    # H/2, W/2
            DepthwiseSeparableConv2d(_ch(64), _ch(128), stride=2),   # H/4, W/4
            DepthwiseSeparableConv2d(_ch(128), _ch(256), stride=2),  # H/8, W/8
            DepthwiseSeparableConv2d(_ch(256), _ch(256), stride=1),  # Same
            DepthwiseSeparableConv2d(_ch(256), _ch(512), stride=2),  # H/16, W/16
        )

        # Adaptive pooling over frequency dimension (keep time for binning)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool freq to 1

        # Project to embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(_ch(512), embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Full forward pass: log-Mel spectrogram → z_a.

        Parameters
        ----------
        mel_spec:
            Tensor of shape ``(B, 1, n_mels, T_spec)`` — log-Mel spectrograms.
            Or ``(B, n_mels, T_spec)`` which will be unsqueezed.

        Returns
        -------
        torch.Tensor
            z_a of shape ``(B, T, embedding_dim)`` where T=8.
        """
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)  # Add channel dim

        B = mel_spec.shape[0]

        # Feature extraction
        x = self.initial_conv(mel_spec)        # (B, 32, n_mels, T_spec)
        x = self.ds_blocks(x)                  # (B, 512, H', W')

        # Pool over frequency, keeping time
        x = self.freq_pool(x)                  # (B, 512, 1, W')
        x = x.squeeze(2)                       # (B, 512, W')
        x = x.permute(0, 2, 1)                # (B, W', 512)

        # Project to embedding_dim
        x = self.projector(x)                  # (B, W', embedding_dim)

        # Pool into T temporal bins
        z_a = self._pool_to_bins(x)            # (B, T, embedding_dim)

        return z_a

    def _pool_to_bins(self, features: torch.Tensor) -> torch.Tensor:
        """Pool time-distributed features into T temporal bins.

        Parameters
        ----------
        features:
            Shape ``(B, L, D)`` where L is the number of time steps.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, D)``.
        """
        B, L, D = features.shape
        T = self.temporal_bins

        if L <= T:
            # Pad if fewer time steps than bins
            if L < T:
                pad = torch.zeros(B, T - L, D, device=features.device, dtype=features.dtype)
                features = torch.cat([features, pad], dim=1)
            return features

        # Assign time steps to bins
        frames_per_bin = L // T
        remainder = L % T

        bins = []
        idx = 0
        for t in range(T):
            n = frames_per_bin + (1 if t < remainder else 0)
            bin_feat = features[:, idx:idx + n, :].mean(dim=1)
            bins.append(bin_feat)
            idx += n

        return torch.stack(bins, dim=1)  # (B, T, D)
