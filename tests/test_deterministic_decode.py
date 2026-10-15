"""Verifies a fixture clip decodes identically across repeated runs given the same seed.

Status: implemented — verifies bit-for-bit identical waveforms, frames, timestamps,
and clipping statistics across repeated decodes of audio and video fixture clips.
See AGENTS.md for build order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.decode.audio_decoder import decode_audio, decode_audio_window
from src.decode.video_decoder import decode_video, decode_video_window
from src.utils.seed_utils import set_seed

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_deterministic_audio_decode() -> None:
    """Repeated decoding of a WAV fixture yields bit-for-bit identical waveforms."""
    wav_path = FIXTURES_DIR / "test_tone.wav"
    if not wav_path.exists():
        pytest.skip("WAV fixture not found")

    set_seed(42)
    res1 = decode_audio(wav_path, target_sr=16_000, peak_normalize=True)

    set_seed(42)
    res2 = decode_audio(wav_path, target_sr=16_000, peak_normalize=True)

    # Waveform arrays must be exactly identical
    np.testing.assert_array_equal(
        res1.waveform,
        res2.waveform,
        err_msg="Decoded waveforms differ across runs",
    )

    # Metadata must match
    assert res1.sample_rate == res2.sample_rate == 16_000
    assert np.isclose(res1.duration_s, res2.duration_s)
    assert np.isclose(res1.clipped_sample_ratio, res2.clipped_sample_ratio)
    assert np.isclose(res1.peak_amplitude, res2.peak_amplitude)


def test_deterministic_audio_window_decode() -> None:
    """Repeated windowed audio decodes produce identical 2s segments."""
    wav_path = FIXTURES_DIR / "test_tone.wav"
    if not wav_path.exists():
        pytest.skip("WAV fixture not found")

    w1 = decode_audio_window(wav_path, start_s=0.0, window_duration_s=1.0)
    w2 = decode_audio_window(wav_path, start_s=0.0, window_duration_s=1.0)

    np.testing.assert_array_equal(w1.waveform, w2.waveform)
    assert len(w1.waveform) == len(w2.waveform)


def test_deterministic_video_decode() -> None:
    """Repeated decoding of a video fixture yields bit-for-bit identical frames and timestamps."""
    npz_path = FIXTURES_DIR / "test_clip.npz"
    if not npz_path.exists():
        pytest.skip("Video fixture not found")

    set_seed(42)
    v1 = decode_video(npz_path, target_fps=25)

    set_seed(42)
    v2 = decode_video(npz_path, target_fps=25)

    np.testing.assert_array_equal(
        v1.frames,
        v2.frames,
        err_msg="Decoded video frames differ across runs",
    )
    np.testing.assert_array_equal(
        v1.timestamps_s,
        v2.timestamps_s,
        err_msg="Decoded video timestamps differ across runs",
    )
    assert v1.target_fps == v2.target_fps == 25
    assert v1.total_native_frames == v2.total_native_frames


def test_video_window_uses_fixed_fps_and_contract_shape() -> None:
    """Window decode returns 16 samples on the fixed 25-fps/2-second time grid."""
    npz_path = FIXTURES_DIR / "test_clip.npz"
    if not npz_path.exists():
        pytest.skip("Video fixture not found")

    window = decode_video_window(
        npz_path, start_s=0.0, window_duration_s=2.0,
        frames_per_window=16, target_fps=25,
    )
    assert window.frames.shape == (16, 160, 160, 3)
    assert window.target_fps == 25
    assert len(window.timestamps_s) == 16
    np.testing.assert_allclose(np.diff(window.timestamps_s), 0.125)
    assert window.dropped_frame_indices == []
