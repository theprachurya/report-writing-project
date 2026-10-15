"""Decodes and resamples the audio track to 16 kHz mono for spectrogram extraction.

Status: implemented — provides decode_audio() for full-track decoding and
decode_audio_window() for extracting a 2-second aligned audio segment.
Peak normalization is applied after recording clipping statistics (Section 6).
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioDecodeResult:
    """Result of decoding an audio track.

    Attributes
    ----------
    waveform:
        Mono waveform as float32 array of shape ``(num_samples,)``.
    sample_rate:
        Output sample rate (16 kHz).
    duration_s:
        Duration in seconds.
    original_sample_rate:
        The native sample rate before resampling.
    clipped_sample_ratio:
        Fraction of samples that were at ±1.0 before peak normalization.
    peak_amplitude:
        Absolute peak amplitude before normalization.
    was_peak_normalized:
        Whether peak normalization was applied.
    """
    waveform: np.ndarray               # (num_samples,) float32
    sample_rate: int
    duration_s: float
    original_sample_rate: int
    clipped_sample_ratio: float
    peak_amplitude: float
    was_peak_normalized: bool


def _load_audio_torchaudio(
    audio_path: Path,
    target_sr: int,
) -> tuple[np.ndarray, int]:
    """Load audio using torchaudio (preferred when available)."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy(), sr


def _load_audio_ffmpeg(
    audio_path: Path,
    target_sr: int,
) -> tuple[np.ndarray, int]:
    """Load audio using ffmpeg subprocess fallback.

    Uses ffmpeg to decode and resample to mono PCM at *target_sr*.
    Returns (waveform, original_sr) where original_sr is probed separately.
    """
    # Probe original sample rate
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, check=True
        )
        original_sr = int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        original_sr = target_sr  # fallback

    # Decode to raw PCM
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(audio_path),
        "-ac", "1",                     # mono
        "-ar", str(target_sr),          # resample
        "-f", "f32le",                  # 32-bit float little-endian PCM
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    waveform = np.frombuffer(proc.stdout, dtype=np.float32)
    return waveform, original_sr


def _load_audio_wavfile(
    audio_path: Path,
    target_sr: int,
) -> tuple[np.ndarray, int]:
    """Load WAV audio using scipy.io.wavfile (standard scipy dependency)."""
    from scipy.io import wavfile
    from scipy import signal

    sr, data = wavfile.read(str(audio_path))
    if data.dtype == np.int16:
        waveform = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        waveform = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        waveform = (data.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = data.astype(np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        num_target = int(round(len(waveform) * target_sr / sr))
        waveform = signal.resample(waveform, num_target).astype(np.float32)

    return waveform, sr


def decode_audio(
    audio_path: str | Path,
    target_sr: int = 16_000,
    peak_normalize: bool = True,
) -> AudioDecodeResult:
    """Decode an audio file (or the audio track of a video) to 16 kHz mono.

    Clipping statistics are recorded *before* peak normalization, matching the
    input contract in Section 6.

    Parameters
    ----------
    audio_path:
        Path to an audio or video file containing an audio track.
    target_sr:
        Target sample rate.  Default 16000 (16 kHz).
    peak_normalize:
        Whether to peak-normalize after recording clipping stats.

    Returns
    -------
    AudioDecodeResult
        Decoded waveform and metadata.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform: np.ndarray | None = None
    original_sr: int = target_sr

    # If it's a WAV file, scipy can decode it directly and reliably
    if audio_path.suffix.lower() == ".wav":
        try:
            waveform, original_sr = _load_audio_wavfile(audio_path, target_sr)
        except Exception as exc:
            logger.debug("scipy wav read failed (%s), trying fallbacks", exc)

    if waveform is None:
        try:
            waveform, original_sr = _load_audio_torchaudio(audio_path, target_sr)
        except (ImportError, RuntimeError) as exc:
            logger.debug("torchaudio unavailable (%s), falling back to ffmpeg", exc)
            waveform, original_sr = _load_audio_ffmpeg(audio_path, target_sr)

    if len(waveform) == 0:
        raise RuntimeError(f"Empty audio track in {audio_path}")

    # Record clipping statistics BEFORE normalization
    peak_amp = float(np.max(np.abs(waveform)))
    clipping_threshold = 0.99  # Samples near ±1.0
    n_clipped = int(np.sum(np.abs(waveform) >= clipping_threshold))
    clipped_ratio = n_clipped / len(waveform)

    if clipped_ratio > 0.01:
        logger.warning(
            "%.1f%% of samples clipped in %s",
            clipped_ratio * 100,
            audio_path.name,
        )

    # Peak normalize (Section 6: "peak-normalize only after recording clipping")
    was_normalized = False
    if peak_normalize and peak_amp > 0:
        waveform = waveform / peak_amp
        was_normalized = True

    duration_s = len(waveform) / target_sr

    return AudioDecodeResult(
        waveform=waveform.astype(np.float32),
        sample_rate=target_sr,
        duration_s=duration_s,
        original_sample_rate=original_sr,
        clipped_sample_ratio=clipped_ratio,
        peak_amplitude=peak_amp,
        was_peak_normalized=was_normalized,
    )


def decode_audio_window(
    audio_path: str | Path,
    start_s: float = 0.0,
    window_duration_s: float = 2.0,
    target_sr: int = 16_000,
    peak_normalize: bool = True,
) -> AudioDecodeResult:
    """Decode a 2-second window of audio matching a video window.

    Parameters
    ----------
    audio_path:
        Path to audio/video file.
    start_s:
        Window start time in seconds.
    window_duration_s:
        Window duration in seconds (default 2.0).
    target_sr:
        Target sample rate (default 16000).
    peak_normalize:
        Whether to peak-normalize after recording clipping.

    Returns
    -------
    AudioDecodeResult
        The windowed audio segment.
    """
    full = decode_audio(
        audio_path, target_sr=target_sr, peak_normalize=False
    )

    start_sample = int(round(start_s * target_sr))
    end_sample = int(round((start_s + window_duration_s) * target_sr))
    end_sample = min(end_sample, len(full.waveform))

    if start_sample >= len(full.waveform):
        raise RuntimeError(
            f"Window start {start_s}s exceeds audio duration "
            f"{full.duration_s:.2f}s"
        )

    waveform = full.waveform[start_sample:end_sample]

    # Pad with zeros if the window extends beyond the audio
    expected_samples = int(round(window_duration_s * target_sr))
    if len(waveform) < expected_samples:
        pad = np.zeros(expected_samples - len(waveform), dtype=np.float32)
        waveform = np.concatenate([waveform, pad])

    # Compute clipping stats for this window
    peak_amp = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
    n_clipped = int(np.sum(np.abs(waveform) >= 0.99))
    clipped_ratio = n_clipped / len(waveform) if len(waveform) > 0 else 0.0

    was_normalized = False
    if peak_normalize and peak_amp > 0:
        waveform = waveform / peak_amp
        was_normalized = True

    return AudioDecodeResult(
        waveform=waveform.astype(np.float32),
        sample_rate=target_sr,
        duration_s=len(waveform) / target_sr,
        original_sample_rate=full.original_sample_rate,
        clipped_sample_ratio=clipped_ratio,
        peak_amplitude=peak_amp,
        was_peak_normalized=was_normalized,
    )
