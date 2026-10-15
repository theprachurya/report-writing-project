"""Applies modality-specific degradations (video codec/resize/blur/frame-drop; audio noise/reverb/low-pass/dropout) — one modality at a time per Section 6.

Status: implemented — provides VideoAugmentor and AudioAugmentor for training-time
modality-specific degradation. Augmentation labels describe quality, never fake/real
status. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationRecord:
    """Records what augmentation was applied (for diagnostics, not labels).

    Attributes
    ----------
    modality:
        Which modality was augmented (``"video"``, ``"audio"``, ``"none"``).
    transform_name:
        Name of the transform applied.
    parameters:
        Transform parameters.
    """
    modality: str
    transform_name: str
    parameters: dict


class VideoAugmentor:
    """Applies video-specific degradations to face crops.

    Each call applies at most one degradation, chosen randomly based on
    configured probabilities.  This matches Section 6: "randomly apply one
    modality at a time."

    Parameters
    ----------
    resize_prob:
        Probability of applying resize degradation.
    blur_prob:
        Probability of applying Gaussian blur.
    frame_drop_prob:
        Probability of dropping (zeroing) random frames.
    reencode_prob:
        Probability of JPEG re-encoding to simulate compression artifacts.
    noise_prob:
        Probability of adding Gaussian noise.
    """

    def __init__(
        self,
        resize_prob: float = 0.3,
        blur_prob: float = 0.2,
        frame_drop_prob: float = 0.1,
        reencode_prob: float = 0.0,
        noise_prob: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.resize_prob = resize_prob
        self.blur_prob = blur_prob
        self.frame_drop_prob = frame_drop_prob
        self.reencode_prob = reencode_prob
        self.noise_prob = noise_prob
        self._rng = rng or np.random.default_rng()

    def __call__(
        self,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Apply a random video augmentation to a window of frames.

        Parameters
        ----------
        frames:
            Array of shape ``(N, H, W, 3)`` uint8 RGB.

        Returns
        -------
        tuple[np.ndarray, AugmentationRecord]
            Augmented frames and a record of what was applied.
        """
        roll = self._rng.random()
        cumulative = 0.0

        # Resize degradation
        cumulative += self.resize_prob
        if roll < cumulative:
            return self._apply_resize(frames)

        # Gaussian blur
        cumulative += self.blur_prob
        if roll < cumulative:
            return self._apply_blur(frames)

        # Frame drop
        cumulative += self.frame_drop_prob
        if roll < cumulative:
            return self._apply_frame_drop(frames)

        cumulative += self.reencode_prob
        if roll < cumulative:
            return self._apply_reencode(frames)

        # Gaussian noise
        cumulative += self.noise_prob
        if roll < cumulative:
            return self._apply_noise(frames)

        # No augmentation
        return frames.copy(), AugmentationRecord("none", "identity", {})

    def _apply_resize(
        self, frames: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Downscale then upscale to simulate resolution loss."""
        h, w = frames.shape[1], frames.shape[2]
        scale = float(self._rng.uniform(0.3, 0.7))
        new_h, new_w = max(8, int(h * scale)), max(8, int(w * scale))

        result = np.empty_like(frames)
        for i in range(len(frames)):
            small = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            result[i] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        return result, AugmentationRecord(
            "video", "resize", {"scale": scale, "intermediate_size": (new_h, new_w)}
        )

    def _apply_blur(
        self, frames: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Apply Gaussian blur."""
        sigma = float(self._rng.uniform(1.0, 4.0))
        ksize = int(sigma * 4) | 1  # Ensure odd kernel size

        result = np.empty_like(frames)
        for i in range(len(frames)):
            result[i] = cv2.GaussianBlur(frames[i], (ksize, ksize), sigma)

        return result, AugmentationRecord(
            "video", "gaussian_blur", {"sigma": sigma, "ksize": ksize}
        )

    def _apply_frame_drop(
        self, frames: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Zero out random frames to simulate frame drops."""
        n_frames = len(frames)
        n_drop = max(1, int(n_frames * self._rng.uniform(0.1, 0.3)))
        drop_indices = self._rng.choice(n_frames, size=n_drop, replace=False).tolist()

        result = frames.copy()
        for idx in drop_indices:
            result[idx] = 0

        return result, AugmentationRecord(
            "video", "frame_drop", {"dropped_indices": drop_indices}
        )

    def _apply_noise(
        self, frames: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Add Gaussian noise."""
        sigma = float(self._rng.uniform(5.0, 25.0))
        noise = self._rng.normal(0, sigma, size=frames.shape).astype(np.float32)
        result = np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return result, AugmentationRecord(
            "video", "gaussian_noise", {"sigma": sigma}
        )

    def _apply_reencode(
        self, frames: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """JPEG re-encode each crop to simulate a lossy compression pass."""
        quality = int(self._rng.integers(35, 86))
        result = np.empty_like(frames)
        for index, frame in enumerate(frames):
            encoded_ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if encoded_ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                result[index] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            else:
                result[index] = frame
        return result, AugmentationRecord("video", "jpeg_reencode", {"quality": quality})


class AudioAugmentor:
    """Applies audio-specific degradations to waveforms.

    Each call applies at most one degradation, matching the "one modality
    at a time" constraint.

    Parameters
    ----------
    noise_prob:
        Probability of adding white noise.
    lowpass_prob:
        Probability of applying a low-pass filter.
    dropout_prob:
        Probability of applying packet-like dropout.
    reverb_prob:
        Probability of adding synthetic reverberation.
    """

    def __init__(
        self,
        noise_prob: float = 0.3,
        lowpass_prob: float = 0.2,
        dropout_prob: float = 0.1,
        reverb_prob: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.noise_prob = noise_prob
        self.lowpass_prob = lowpass_prob
        self.dropout_prob = dropout_prob
        self.reverb_prob = reverb_prob
        self._rng = rng or np.random.default_rng()

    def __call__(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16_000,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Apply a random audio augmentation.

        Parameters
        ----------
        waveform:
            1-D float32 mono waveform.
        sample_rate:
            Sample rate in Hz.

        Returns
        -------
        tuple[np.ndarray, AugmentationRecord]
            Augmented waveform and record.
        """
        roll = self._rng.random()
        cumulative = 0.0

        cumulative += self.noise_prob
        if roll < cumulative:
            return self._apply_awgn(waveform)

        cumulative += self.lowpass_prob
        if roll < cumulative:
            return self._apply_lowpass(waveform, sample_rate)

        cumulative += self.dropout_prob
        if roll < cumulative:
            return self._apply_dropout(waveform, sample_rate)

        cumulative += self.reverb_prob
        if roll < cumulative:
            return self._apply_reverb(waveform, sample_rate)

        return waveform.copy(), AugmentationRecord("none", "identity", {})

    def _apply_awgn(
        self, waveform: np.ndarray,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Add additive white Gaussian noise at a random SNR."""
        snr_db = float(self._rng.uniform(0, 20))
        signal_power = float(np.mean(waveform ** 2))
        if signal_power < 1e-10:
            return waveform.copy(), AugmentationRecord(
                "audio", "awgn", {"snr_db": snr_db, "skipped": True}
            )
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self._rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
        result = waveform + noise

        return result, AugmentationRecord(
            "audio", "awgn", {"snr_db": snr_db}
        )

    def _apply_lowpass(
        self, waveform: np.ndarray, sample_rate: int,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Apply a simple low-pass filter."""
        cutoff_hz = float(self._rng.uniform(1000, 4000))
        try:
            from scipy.signal import butter, sosfilt
            nyq = sample_rate / 2.0
            normalized_cutoff = min(cutoff_hz / nyq, 0.99)
            sos = butter(4, normalized_cutoff, btype="low", output="sos")
            result = sosfilt(sos, waveform).astype(np.float32)
        except ImportError:
            # Fallback: simple moving average
            kernel_size = max(3, int(sample_rate / cutoff_hz))
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            result = np.convolve(waveform, kernel, mode="same").astype(np.float32)

        return result, AugmentationRecord(
            "audio", "lowpass", {"cutoff_hz": cutoff_hz}
        )

    def _apply_dropout(
        self, waveform: np.ndarray, sample_rate: int,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Simulate packet-like dropout by zeroing random segments."""
        n_segments = int(self._rng.integers(1, 5))
        segment_ms = float(self._rng.uniform(10, 50))
        segment_samples = int(segment_ms * sample_rate / 1000)

        result = waveform.copy()
        dropped_ranges: list[tuple[int, int]] = []
        for _ in range(n_segments):
            start = int(self._rng.integers(0, max(1, len(waveform) - segment_samples)))
            end = min(start + segment_samples, len(waveform))
            result[start:end] = 0.0
            dropped_ranges.append((start, end))

        return result, AugmentationRecord(
            "audio", "packet_dropout",
            {"n_segments": n_segments, "segment_ms": segment_ms},
        )

    def _apply_reverb(
        self, waveform: np.ndarray, sample_rate: int,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Add simple synthetic reverberation using a comb filter."""
        delay_ms = float(self._rng.uniform(10, 50))
        decay = float(self._rng.uniform(0.2, 0.6))
        delay_samples = int(delay_ms * sample_rate / 1000)

        result = waveform.copy()
        for i in range(delay_samples, len(result)):
            result[i] += decay * result[i - delay_samples]

        # Normalize to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak

        return result.astype(np.float32), AugmentationRecord(
            "audio", "reverb", {"delay_ms": delay_ms, "decay": decay}
        )


def apply_modality_dropout(
    visual_features: np.ndarray,
    audio_features: np.ndarray,
    prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Randomly zero out one modality's features (modality dropout).

    This trains the model to handle missing modalities gracefully.

    Parameters
    ----------
    visual_features:
        Visual feature tensor.
    audio_features:
        Audio feature tensor.
    prob:
        Probability of dropping a modality.
    rng:
        Random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, str]
        (visual, audio, dropped_modality) where ``dropped_modality`` is
        ``"visual"``, ``"audio"``, or ``"none"``.
    """
    rng = rng or np.random.default_rng()
    if rng.random() < prob:
        if rng.random() < 0.5:
            return np.zeros_like(visual_features), audio_features, "visual"
        else:
            return visual_features, np.zeros_like(audio_features), "audio"
    return visual_features, audio_features, "none"
