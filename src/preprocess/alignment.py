"""Bins visual and audio encoder outputs into T=8 temporal bins per 2s window and stores the timestamp transform.

Status: implemented — provides TemporalAligner for mapping raw frame-level and
spectrogram-level features into T=8 aligned temporal bins, each producing a single
embedding vector. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of temporal alignment.

    Attributes
    ----------
    visual_bins:
        Visual features pooled into T bins, shape ``(T, D)`` or list of frame
        indices per bin.
    audio_bins:
        Audio features pooled into T bins, shape ``(T, D)`` or list of
        spectrogram frame ranges per bin.
    timestamp_transform:
        Mapping from bin index to (start_s, end_s) time range.
    n_bins:
        Number of temporal bins.
    dropped_frames:
        Frames that couldn't be assigned to any bin.
    """
    visual_bins: np.ndarray | list[list[int]]
    audio_bins: np.ndarray | list[tuple[int, int]]
    timestamp_transform: list[tuple[float, float]]
    n_bins: int
    dropped_frames: list[int] = field(default_factory=list)


class TemporalAligner:
    """Aligns visual and audio features into T temporal bins per window.

    Per Section 6: divide each 2 s window into T=8 temporal bins.  Pool visual
    and audio encoder outputs to one embedding vector per bin.

    Parameters
    ----------
    n_bins:
        Number of temporal bins (default 8).
    window_duration_s:
        Window duration in seconds (default 2.0).
    """

    def __init__(
        self,
        n_bins: int = 8,
        window_duration_s: float = 2.0,
    ) -> None:
        self.n_bins = n_bins
        self.window_duration_s = window_duration_s
        self.bin_duration_s = window_duration_s / n_bins

    def compute_timestamp_transform(self) -> list[tuple[float, float]]:
        """Compute the (start_s, end_s) range for each temporal bin.

        Returns
        -------
        list[tuple[float, float]]
            List of T ``(start_s, end_s)`` tuples.
        """
        return [
            (i * self.bin_duration_s, (i + 1) * self.bin_duration_s)
            for i in range(self.n_bins)
        ]

    def assign_frames_to_bins(
        self,
        timestamps_s: np.ndarray,
        start_s: float = 0.0,
    ) -> list[list[int]]:
        """Assign frame indices to temporal bins based on their timestamps.

        Parameters
        ----------
        timestamps_s:
            Per-frame timestamps in seconds (relative to video start).
        start_s:
            Start time of the window.

        Returns
        -------
        list[list[int]]
            For each bin, the list of frame indices falling in that bin.
        """
        bins: list[list[int]] = [[] for _ in range(self.n_bins)]
        for idx, ts in enumerate(timestamps_s):
            relative_ts = ts - start_s
            bin_idx = int(relative_ts / self.bin_duration_s)
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))
            bins[bin_idx].append(idx)
        return bins

    def assign_spectrogram_frames_to_bins(
        self,
        n_spec_frames: int,
        hop_length: int,
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        """Map spectrogram time frames to temporal bins.

        Parameters
        ----------
        n_spec_frames:
            Total number of spectrogram time frames.
        hop_length:
            Spectrogram hop length in samples.
        sample_rate:
            Audio sample rate.

        Returns
        -------
        list[tuple[int, int]]
            For each bin, ``(start_frame, end_frame)`` slice indices into
            the spectrogram.
        """
        total_duration = n_spec_frames * hop_length / sample_rate
        bins: list[tuple[int, int]] = []
        for i in range(self.n_bins):
            bin_start_s = i * self.bin_duration_s
            bin_end_s = (i + 1) * self.bin_duration_s
            start_frame = int(bin_start_s * sample_rate / hop_length)
            end_frame = int(bin_end_s * sample_rate / hop_length)
            start_frame = max(0, min(start_frame, n_spec_frames))
            end_frame = max(start_frame, min(end_frame, n_spec_frames))
            bins.append((start_frame, end_frame))
        return bins

    def pool_visual_features(
        self,
        frame_features: np.ndarray,
        frame_bins: list[list[int]],
    ) -> np.ndarray:
        """Pool per-frame visual features into temporal bin embeddings.

        Parameters
        ----------
        frame_features:
            Per-frame feature array of shape ``(N_frames, D)``.
        frame_bins:
            Frame-to-bin assignments from :meth:`assign_frames_to_bins`.

        Returns
        -------
        np.ndarray
            Pooled features of shape ``(T, D)``.  Empty bins get zeros.
        """
        d = frame_features.shape[1] if frame_features.ndim > 1 else frame_features.shape[0]
        pooled = np.zeros((self.n_bins, d), dtype=np.float32)

        for bin_idx, frame_indices in enumerate(frame_bins):
            if frame_indices:
                valid_indices = [i for i in frame_indices if i < len(frame_features)]
                if valid_indices:
                    pooled[bin_idx] = frame_features[valid_indices].mean(axis=0)

        return pooled

    def pool_audio_features(
        self,
        spectrogram_features: np.ndarray,
        spec_bins: list[tuple[int, int]],
    ) -> np.ndarray:
        """Pool spectrogram features into temporal bin embeddings.

        Parameters
        ----------
        spectrogram_features:
            Feature array of shape ``(D, T_spec)`` (Mel bins × time frames)
            or ``(T_spec, D)`` if already transposed.
        spec_bins:
            Spectrogram-to-bin assignments from
            :meth:`assign_spectrogram_frames_to_bins`.

        Returns
        -------
        np.ndarray
            Pooled features of shape ``(T, D)``.
        """
        # Ensure shape is (T_spec, D) for pooling
        if spectrogram_features.ndim == 2 and spectrogram_features.shape[0] < spectrogram_features.shape[1]:
            spec = spectrogram_features.T  # (T_spec, n_mels)
        else:
            spec = spectrogram_features

        d = spec.shape[1] if spec.ndim > 1 else spec.shape[0]
        pooled = np.zeros((self.n_bins, d), dtype=np.float32)

        for bin_idx, (start, end) in enumerate(spec_bins):
            if end > start and start < len(spec):
                end = min(end, len(spec))
                pooled[bin_idx] = spec[start:end].mean(axis=0)

        return pooled

    def align(
        self,
        visual_frame_features: np.ndarray,
        visual_timestamps_s: np.ndarray,
        audio_spectrogram: np.ndarray,
        audio_hop_length: int,
        audio_sample_rate: int,
        window_start_s: float = 0.0,
    ) -> AlignmentResult:
        """Full alignment pipeline: assign and pool both modalities.

        Parameters
        ----------
        visual_frame_features:
            Per-frame visual features ``(N_frames, D)``.
        visual_timestamps_s:
            Per-frame timestamps in seconds.
        audio_spectrogram:
            Log-Mel spectrogram ``(n_mels, T_spec)``.
        audio_hop_length:
            Spectrogram hop length in samples.
        audio_sample_rate:
            Audio sample rate.
        window_start_s:
            Start time of the window.

        Returns
        -------
        AlignmentResult
            Aligned visual and audio bin features with timestamp transform.
        """
        timestamp_transform = self.compute_timestamp_transform()

        # Visual alignment
        frame_bins = self.assign_frames_to_bins(
            visual_timestamps_s, start_s=window_start_s
        )
        visual_pooled = self.pool_visual_features(
            visual_frame_features, frame_bins
        )

        # Audio alignment
        n_spec_frames = audio_spectrogram.shape[1] if audio_spectrogram.ndim == 2 else len(audio_spectrogram)
        spec_bins = self.assign_spectrogram_frames_to_bins(
            n_spec_frames, audio_hop_length, audio_sample_rate
        )
        audio_pooled = self.pool_audio_features(audio_spectrogram, spec_bins)

        # Track frames that got empty bins
        empty_bins = [i for i, fb in enumerate(frame_bins) if not fb]
        if empty_bins:
            logger.debug("Empty visual bins: %s", empty_bins)

        return AlignmentResult(
            visual_bins=visual_pooled,
            audio_bins=audio_pooled,
            timestamp_transform=timestamp_transform,
            n_bins=self.n_bins,
            dropped_frames=empty_bins,
        )
