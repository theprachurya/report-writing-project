"""Verifies visual/audio temporal bin alignment matches the stored timestamp transform.

Status: implemented — tests timestamp intervals, frame assignment, spectrogram
slicing, and feature pooling per Section 6. See AGENTS.md for build order.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocess.alignment import AlignmentResult, TemporalAligner


def test_timestamp_transform_intervals() -> None:
    """Verifies that the timestamp transform covers [0, duration] with T contiguous bins."""
    aligner = TemporalAligner(n_bins=8, window_duration_s=2.0)
    transform = aligner.compute_timestamp_transform()

    assert len(transform) == 8, f"Expected 8 bins, got {len(transform)}"

    # First bin starts at 0.0, last bin ends at 2.0
    assert np.isclose(transform[0][0], 0.0)
    assert np.isclose(transform[-1][1], 2.0)

    # Check contiguity and non-overlapping bin duration (0.25s each)
    expected_duration = 0.25
    for i, (start, end) in enumerate(transform):
        assert np.isclose(end - start, expected_duration), (
            f"Bin {i} duration is {end - start}, expected {expected_duration}"
        )
        if i > 0:
            assert np.isclose(transform[i - 1][1], start), (
                f"Gap or overlap between bin {i - 1} and {i}"
            )


def test_assign_frames_to_bins() -> None:
    """Verifies visual frame timestamps map to the correct temporal bins."""
    aligner = TemporalAligner(n_bins=8, window_duration_s=2.0)
    transform = aligner.compute_timestamp_transform()

    # 16 frames uniformly spaced across [0, 2] s (at ~8 effective visual fps)
    timestamps = np.array([
        0.05, 0.15,  # bin 0: [0.0, 0.25)
        0.30, 0.40,  # bin 1: [0.25, 0.50)
        0.55, 0.65,  # bin 2: [0.50, 0.75)
        0.80, 0.90,  # bin 3: [0.75, 1.00)
        1.05, 1.15,  # bin 4: [1.00, 1.25)
        1.30, 1.40,  # bin 5: [1.25, 1.50)
        1.55, 1.65,  # bin 6: [1.50, 1.75)
        1.80, 1.95,  # bin 7: [1.75, 2.00)
    ])

    frame_bins = aligner.assign_frames_to_bins(timestamps, start_s=0.0)

    assert len(frame_bins) == 8
    # Each bin should receive exactly 2 frames in this uniform arrangement
    for b_idx in range(8):
        assert len(frame_bins[b_idx]) == 2
        for frame_idx in frame_bins[b_idx]:
            ts = timestamps[frame_idx]
            bin_start, bin_end = transform[b_idx]
            assert bin_start <= ts <= bin_end, (
                f"Frame {frame_idx} (ts={ts}) misplaced in bin {b_idx} [{bin_start}, {bin_end}]"
            )


def test_assign_spectrogram_frames_to_bins() -> None:
    """Verifies audio spectrogram frame ranges match bin time boundaries."""
    sample_rate = 16_000
    hop_length = 160  # 10 ms hop -> 100 frames/sec
    duration_s = 2.0
    n_spec_frames = int(duration_s * sample_rate / hop_length)  # 200 frames

    aligner = TemporalAligner(n_bins=8, window_duration_s=duration_s)
    transform = aligner.compute_timestamp_transform()
    spec_bins = aligner.assign_spectrogram_frames_to_bins(
        n_spec_frames=n_spec_frames,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )

    assert len(spec_bins) == 8

    # Each bin should correspond to 0.25 s = 25 frames
    for b_idx, (start_f, end_f) in enumerate(spec_bins):
        bin_start_s, bin_end_s = transform[b_idx]
        expected_start = int(bin_start_s * sample_rate / hop_length)
        expected_end = int(bin_end_s * sample_rate / hop_length)
        assert start_f == expected_start, f"Bin {b_idx} start mismatch"
        assert end_f == expected_end, f"Bin {b_idx} end mismatch"
        assert end_f - start_f == 25


def test_full_alignment_pipeline() -> None:
    """Verifies end-to-end alignment output shapes and contents."""
    aligner = TemporalAligner(n_bins=8, window_duration_s=2.0)

    n_frames = 16
    d_v = 256
    n_mels = 80
    n_spec_frames = 200

    visual_features = np.random.randn(n_frames, d_v).astype(np.float32)
    visual_timestamps = np.linspace(0.06, 1.94, n_frames)
    audio_spec = np.random.randn(n_mels, n_spec_frames).astype(np.float32)

    result = aligner.align(
        visual_frame_features=visual_features,
        visual_timestamps_s=visual_timestamps,
        audio_spectrogram=audio_spec,
        audio_hop_length=160,
        audio_sample_rate=16_000,
        window_start_s=0.0,
    )

    assert isinstance(result, AlignmentResult)
    assert result.n_bins == 8
    assert result.visual_bins.shape == (8, d_v)
    assert result.audio_bins.shape == (8, n_mels)
    assert len(result.timestamp_transform) == 8
    assert result.timestamp_transform == aligner.compute_timestamp_transform()


def test_alignment_with_custom_window_and_bins() -> None:
    """Verifies temporal alignment with arbitrary T and duration."""
    aligner = TemporalAligner(n_bins=4, window_duration_s=1.0)
    transform = aligner.compute_timestamp_transform()

    assert len(transform) == 4
    assert np.isclose(transform[0][0], 0.0)
    assert np.isclose(transform[-1][1], 1.0)
    assert np.isclose(aligner.bin_duration_s, 0.25)
