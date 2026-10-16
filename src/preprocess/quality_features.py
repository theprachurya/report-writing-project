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
        Mean luminance (grayscale intensity / brightness) across face crops.
    luminance_std:
        Standard deviation of luminance across face crops.
    saturation_mean:
        Mean saturation (HSV S-channel) across face crops.
    failed_frame_fraction:
        Fraction of frames where face detection failed.
    detection_confidence_mean:
        Mean detector confidence score across successful detections.
    """
    detection_success_rate: float
    blur_variance: float
    face_area_mean: float
    luminance_mean: float
    luminance_std: float
    saturation_mean: float
    failed_frame_fraction: float
    detection_confidence_mean: float = 0.0

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
        List of face crops (RGB uint8 or float tensors/arrays).
        ``None`` entries indicate failed detections.
    face_areas:
        Per-frame face bounding-box areas. If ``None``, computed from crops.
    detection_confidences:
        Per-frame detector confidences.

    Returns
    -------
    VisualQualityFeatures
        The 7-D quality feature set with confidence metadata.
    """
    n_total = len(crops)
    valid_crops = [c for c in crops if c is not None]
    n_detected = len(valid_crops)
    n_failed = n_total - n_detected

    detection_rate = n_detected / n_total if n_total > 0 else 0.0
    failed_fraction = n_failed / n_total if n_total > 0 else 1.0

    if detection_confidences is not None:
        valid_confs = [float(c) for c in detection_confidences if c > 0]
        conf_mean = float(np.mean(valid_confs)) if valid_confs else 0.0
    else:
        conf_mean = 1.0 if n_detected > 0 else 0.0

    if n_detected == 0:
        return VisualQualityFeatures(
            detection_success_rate=0.0,
            blur_variance=0.0,
            face_area_mean=0.0,
            luminance_mean=0.0,
            luminance_std=0.0,
            saturation_mean=0.0,
            failed_frame_fraction=1.0,
            detection_confidence_mean=0.0,
        )

    # Blur: variance of Laplacian (higher = sharper)
    blur_values = []
    luminances = []
    saturations = []
    for crop in valid_crops:
        # Convert tensor to numpy array before OpenCV operations
        if hasattr(crop, "detach"):
            crop_np = crop.detach().cpu().numpy()
        else:
            crop_np = np.asarray(crop)

        # Handle CHW vs HWC layout if necessary
        if crop_np.ndim == 3 and crop_np.shape[0] in (1, 3) and crop_np.shape[-1] not in (1, 3):
            crop_np = np.transpose(crop_np, (1, 2, 0))

        # Convert float/normalized crops to uint8 RGB for OpenCV
        if np.issubdtype(crop_np.dtype, np.floating):
            if crop_np.min() < 0.0:
                # Approximate de-normalization if ImageNet normalized
                crop_np = np.clip((crop_np * 0.225 + 0.45) * 255.0, 0, 255).astype(np.uint8)
            elif crop_np.max() <= 1.0:
                crop_np = np.clip(crop_np * 255.0, 0, 255).astype(np.uint8)
            else:
                crop_np = np.clip(crop_np, 0, 255).astype(np.uint8)
        else:
            crop_np = crop_np.astype(np.uint8)

        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_values.append(lap_var)

        luminances.append(float(gray.mean()))

        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
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
        detection_confidence_mean=conf_mean,
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

    Vectorized implementation over NumPy arrays for efficiency.

    Parameters
    ----------
    waveform:
        1-D float32 mono waveform (or torch.Tensor).
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
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32).ravel()

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

    # Vectorized energy-based VAD and SNR proxy
    frame_size = int(sample_rate * vad_frame_ms / 1000)
    n_frames = len(waveform) // frame_size

    if n_frames == 0:
        frame_powers = np.array([float(np.mean(waveform ** 2))], dtype=np.float32)
    else:
        truncated_len = n_frames * frame_size
        frames_mat = waveform[:truncated_len].reshape(n_frames, frame_size)
        frame_powers = np.mean(frames_mat ** 2, axis=1)

    frame_energies = np.sqrt(frame_powers)
    active_mask = frame_energies > vad_energy_threshold
    vad_fraction = float(np.mean(active_mask))

    signal_powers = frame_powers[active_mask]
    noise_powers = frame_powers[~active_mask]

    if len(signal_powers) > 0 and len(noise_powers) > 0:
        mean_signal = float(np.mean(signal_powers))
        mean_noise = float(np.mean(noise_powers))
        if mean_noise > 1e-10:
            snr_proxy = float(10.0 * np.log10(mean_signal / mean_noise))
        else:
            snr_proxy = 60.0  # Very clean signal
    elif len(signal_powers) > 0:
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
