"""Computes 80-bin log-Mel spectrograms (25ms window / 10ms hop) from resampled audio.

Status: implemented — provides compute_log_mel_spectrogram() for converting a 16 kHz
mono waveform into an 80-bin log-Mel spectrogram using either torchaudio or a NumPy/scipy
fallback. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MelSpectrogramResult:
    """Result of log-Mel spectrogram computation.

    Attributes
    ----------
    spectrogram:
        Log-Mel spectrogram of shape ``(n_mels, T)`` float32.
    n_mels:
        Number of Mel filter banks.
    sample_rate:
        Audio sample rate used.
    hop_length:
        Hop length in samples.
    window_length:
        Window length in samples.
    n_frames:
        Number of time frames in the spectrogram.
    duration_s:
        Duration of the source audio in seconds.
    """
    spectrogram: np.ndarray   # (n_mels, T) float32
    n_mels: int
    sample_rate: int
    hop_length: int
    window_length: int
    n_frames: int
    duration_s: float


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    n_mels: int = 80,
    window_length_ms: int = 25,
    hop_length_ms: int = 10,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    log_offset: float = 1e-6,
) -> MelSpectrogramResult:
    """Compute an 80-bin log-Mel spectrogram from a mono waveform.

    Parameters
    ----------
    waveform:
        1-D float32 array of mono audio samples at *sample_rate*.
    sample_rate:
        Sample rate in Hz (default 16000).
    n_mels:
        Number of Mel filter banks (default 80).
    window_length_ms:
        STFT window length in milliseconds (default 25).
    hop_length_ms:
        STFT hop length in milliseconds (default 10).
    f_min:
        Minimum frequency for Mel filter bank (Hz).
    f_max:
        Maximum frequency for Mel filter bank (Hz).
    log_offset:
        Small constant added before log to avoid log(0).

    Returns
    -------
    MelSpectrogramResult
        The computed log-Mel spectrogram and metadata.
    """
    win_length = int(sample_rate * window_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)

    # Try torchaudio first for GPU-friendly & exact implementation
    try:
        log_mel = _compute_torchaudio(
            waveform, sample_rate, n_mels, win_length, hop_length, f_min, f_max, log_offset
        )
    except (ImportError, RuntimeError, OSError, Exception) as exc:
        logger.debug("torchaudio failed (%s), falling back to numpy", exc)
        raw_waveform = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        mel_spec = _compute_numpy(
            raw_waveform, sample_rate, n_mels, win_length, hop_length, f_min, f_max
        )
        # Apply log scaling in float32 precision
        log_mel = np.log(mel_spec.astype(np.float32) + log_offset).astype(np.float32)

    return MelSpectrogramResult(
        spectrogram=log_mel,
        n_mels=n_mels,
        sample_rate=sample_rate,
        hop_length=hop_length,
        window_length=win_length,
        n_frames=log_mel.shape[1],
        duration_s=len(waveform) / sample_rate,
    )


def _compute_torchaudio(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    log_offset: float = 1e-6,
) -> np.ndarray:
    """Compute log-Mel spectrogram using torchaudio with float32 precision."""
    import torch
    import torchaudio

    if hasattr(waveform, "detach"):
        tensor = waveform.detach().float()
    else:
        tensor = torch.from_numpy(np.asarray(waveform)).float()

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # (1, T)

    n_fft = max(win_length, 512)  # At least 512 for reasonable frequency resolution

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
    )
    mel = transform(tensor.float())  # (1, n_mels, T)
    # Ensure float32 precision for torch.log to prevent float16 underflow/NaN
    log_mel = torch.log(mel.float() + log_offset)
    return log_mel.squeeze(0).cpu().numpy()


def _compute_numpy(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Compute Mel spectrogram using NumPy/SciPy (fallback)."""
    from scipy import signal as sig

    n_fft = max(win_length, 512)

    # STFT
    _, _, Zxx = sig.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
    )
    power = np.abs(Zxx) ** 2  # (n_freq, T)

    # Mel filter bank
    mel_fb = _mel_filterbank(
        n_mels, n_fft, sample_rate, f_min, f_max
    )  # (n_mels, n_freq)

    mel_spec = mel_fb @ power  # (n_mels, T)
    return mel_spec.astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Create a Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1).

    Ensures no degenerate zero-width filter bins by strictly ordering support points.
    """
    n_freq = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Enforce strictly increasing bin points to prevent degenerate zero-width bins
    bin_points = np.clip(bin_points, 0, n_freq - 1)
    for k in range(1, len(bin_points)):
        if bin_points[k] <= bin_points[k - 1]:
            bin_points[k] = min(bin_points[k - 1] + 1, n_freq - 1)
    for k in range(len(bin_points) - 2, -1, -1):
        if bin_points[k] >= bin_points[k + 1] and bin_points[k + 1] > 0:
            bin_points[k] = bin_points[k + 1] - 1

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        left = int(bin_points[i])
        center = int(bin_points[i + 1])
        right = int(bin_points[i + 2])

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_freq)
        center = min(center, n_freq - 1)
        left = max(0, min(left, center - 1))

        # Rising slope
        denom_up = max(1, center - left)
        for j in range(left, center):
            fb[i, j] = (j - left) / denom_up
        # Falling slope
        denom_down = max(1, right - center)
        for j in range(center, right):
            fb[i, j] = (right - j) / denom_down

        # Degenerate fallback: if filter bank row is all zero, assign center bin
        if fb[i].sum() == 0:
            fb[i, min(center, n_freq - 1)] = 1.0

    return fb
