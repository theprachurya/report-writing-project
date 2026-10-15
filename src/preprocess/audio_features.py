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
        mel_spec = _compute_torchaudio(
            waveform, sample_rate, n_mels, win_length, hop_length, f_min, f_max
        )
    except ImportError:
        mel_spec = _compute_numpy(
            waveform, sample_rate, n_mels, win_length, hop_length, f_min, f_max
        )

    # Apply log scaling
    log_mel = np.log(mel_spec + log_offset).astype(np.float32)

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
) -> np.ndarray:
    """Compute Mel spectrogram using torchaudio."""
    import torch
    import torchaudio

    tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
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
    mel = transform(tensor)  # (1, n_mels, T)
    return mel.squeeze(0).numpy()


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
    """Create a Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1)."""
    n_freq = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        # Rising slope
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        # Falling slope
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)
    return fb
