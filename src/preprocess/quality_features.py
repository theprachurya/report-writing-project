"""Computes transparent modality-quality proxies: visual blur/luminance/detector-confidence; audio RMS/VAD/clipping/SNR proxy.

Status: implemented — provides compute_visual_quality() and compute_audio_quality()
returning the exact quality features listed in Section 7.1 for use by the reliability
estimator. Quality does not equal authenticity.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Visual quality features (Section 7.1) ────────────────────────────────────

@dataclass
class VisualQualityFeatures:
    """Per-window visual quality features (7 dimensions).

    Attributes
    ----------
    detection_success_rate:
        Fraction of frames with a successful face detection.
    blur_variance:
        Mean variance of Laplacian across detected face crops (blur proxy).
    face_area_mean:
        Mean face bounding-box area (pixels) across detections.
    luminance_mean:
        Mean luminance (grayscale intensity) across face crops.
    luminance_std:
        Standard deviation of luminance across face crops.
    saturation_mean:
        Mean saturation (HSV S-channel) across face crops.
    failed_frame_fraction:
        Fraction of frames where face detection failed.
    """
    detection_success_rate: float
    blur_variance: float
    face_area_mean: float
    luminance_mean: float
    luminance_std: float
    saturation_mean: float
    failed_frame_fraction: float

    def to_vector(self) -> np.ndarray:
        """Return features as a 7-D float32 vector."""
        return np.array([
            self.detection_success_rate,
            self.blur_variance,
            self.face_area_mean,
            self.luminance_mean,
            self.luminance_std,
            self.saturation_mean,
            self.failed_frame_fraction,
        ], dtype=np.float32)


def compute_visual_quality(
    crops: list[np.ndarray | None],
    face_areas: list[float] | None = None,
    detection_confidences: list[float] | None = None,
) -> VisualQualityFeatures:
    """Compute visual quality features from a window of face crops.

    Parameters
    ----------
    crops:
        List of face crops (RGB uint8, ``crop_size × crop_size × 3``).
        ``None`` entries indicate failed detections.
    face_areas:
        Per-frame face bounding-box areas.  If ``None``, computed from crops.
    detection_confidences:
        Per-frame detector confidences.  Not directly used in the feature
        vector but logged for diagnostics.

    Returns
    -------
    VisualQualityFeatures
        The 7-D quality feature set.
    """
    n_total = len(crops)
    valid_crops = [c for c in crops if c is not None]
    n_detected = len(valid_crops)
    n_failed = n_total - n_detected

    detection_rate = n_detected / n_total if n_total > 0 else 0.0
    failed_fraction = n_failed / n_total if n_total > 0 else 1.0

    if n_detected == 0:
        return VisualQualityFeatures(
            detection_success_rate=0.0,
            blur_variance=0.0,
            face_area_mean=0.0,
            luminance_mean=0.0,
            luminance_std=0.0,
            saturation_mean=0.0,
            failed_frame_fraction=1.0,
        )

    # Blur: variance of Laplacian (higher = sharper)
    blur_values = []
    luminances = []
    saturations = []
    for crop in valid_crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_values.append(lap_var)

        luminances.append(float(gray.mean()))

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        saturations.append(float(hsv[:, :, 1].mean()))

    # Face area
    if face_areas is not None:
        valid_areas = [a for a in face_areas if a > 0]
        area_mean = float(np.mean(valid_areas)) if valid_areas else 0.0
    else:
        # Estimate from crop sizes (all are crop_size × crop_size after resize)
        area_mean = float(valid_crops[0].shape[0] * valid_crops[0].shape[1])

    return VisualQualityFeatures(
        detection_success_rate=detection_rate,
        blur_variance=float(np.mean(blur_values)),
        face_area_mean=area_mean,
        luminance_mean=float(np.mean(luminances)),
        luminance_std=float(np.std(luminances)),
        saturation_mean=float(np.mean(saturations)),
        failed_frame_fraction=failed_fraction,
    )


# ── Audio quality features (Section 7.1) ─────────────────────────────────────

@dataclass
class AudioQualityFeatures:
    """Per-window audio quality features (5 dimensions).

    Attributes
    ----------
    vad_fraction:
        Fraction of the window that contains voice activity.
    rms:
        Root mean square energy of the waveform.
    clipped_sample_ratio:
        Fraction of samples near ±1.0 (clipping indicator).
    snr_proxy:
        Estimated signal-to-noise ratio proxy in dB.
    duration_mismatch:
        Mismatch between actual audio duration and expected window
        duration (seconds).  Zero if perfectly matched.
    """
    vad_fraction: float
    rms: float
    clipped_sample_ratio: float
    snr_proxy: float
    duration_mismatch: float

    def to_vector(self) -> np.ndarray:
        """Return features as a 5-D float32 vector."""
        return np.array([
            self.vad_fraction,
            self.rms,
            self.clipped_sample_ratio,
            self.snr_proxy,
            self.duration_mismatch,
        ], dtype=np.float32)


def compute_audio_quality(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    expected_duration_s: float = 2.0,
    vad_energy_threshold: float = 0.01,
    vad_frame_ms: int = 25,
    clipping_threshold: float = 0.99,
) -> AudioQualityFeatures:
    """Compute audio quality features from a mono waveform.

    Parameters
    ----------
    waveform:
        1-D float32 mono waveform.
    sample_rate:
        Sample rate (default 16000).
    expected_duration_s:
        Expected window duration for mismatch calculation.
    vad_energy_threshold:
        Energy threshold for voice activity detection.
    vad_frame_ms:
        Frame size in ms for VAD energy estimation.
    clipping_threshold:
        Threshold for considering a sample as clipped.

    Returns
    -------
    AudioQualityFeatures
        The 5-D quality feature set.
    """
    if len(waveform) == 0:
        return AudioQualityFeatures(
            vad_fraction=0.0,
            rms=0.0,
            clipped_sample_ratio=0.0,
            snr_proxy=0.0,
            duration_mismatch=expected_duration_s,
        )

    # RMS energy
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    # Clipping ratio
    n_clipped = int(np.sum(np.abs(waveform) >= clipping_threshold))
    clipped_ratio = n_clipped / len(waveform)

    # Simple energy-based VAD
    frame_size = int(sample_rate * vad_frame_ms / 1000)
    n_frames = max(1, len(waveform) // frame_size)
    active_frames = 0
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(waveform))
        frame_energy = float(np.sqrt(np.mean(waveform[start:end] ** 2)))
        if frame_energy > vad_energy_threshold:
            active_frames += 1
    vad_fraction = active_frames / n_frames

    # SNR proxy: ratio of signal power in active frames to noise power in silent frames
    signal_powers = []
    noise_powers = []
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(waveform))
        frame_power = float(np.mean(waveform[start:end] ** 2))
        frame_energy = float(np.sqrt(frame_power))
        if frame_energy > vad_energy_threshold:
            signal_powers.append(frame_power)
        else:
            noise_powers.append(frame_power)

    if signal_powers and noise_powers:
        mean_signal = float(np.mean(signal_powers))
        mean_noise = float(np.mean(noise_powers))
        if mean_noise > 1e-10:
            snr_proxy = float(10.0 * np.log10(mean_signal / mean_noise))
        else:
            snr_proxy = 60.0  # Very clean signal
    elif signal_powers:
        snr_proxy = 60.0  # No detectable noise frames
    else:
        snr_proxy = 0.0  # No signal detected

    # Duration mismatch
    actual_duration = len(waveform) / sample_rate
    duration_mismatch = abs(actual_duration - expected_duration_s)

    return AudioQualityFeatures(
        vad_fraction=vad_fraction,
        rms=rms,
        clipped_sample_ratio=clipped_ratio,
        snr_proxy=snr_proxy,
        duration_mismatch=duration_mismatch,
    )
