"""Unit tests for transparent modality-quality proxies (src/preprocess/quality_features.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.preprocess.quality_features import (
    AudioQualityFeatures,
    VisualQualityFeatures,
    compute_audio_quality,
    compute_visual_quality,
)


def test_visual_quality_features_standard_crops() -> None:
    """Standard 160x160 RGB crops produce 7-D quality vector within valid ranges."""
    # Synthetic face crop: sharp gradient
    crop1 = np.zeros((160, 160, 3), dtype=np.uint8)
    crop1[:, :80] = 200
    crop2 = np.ones((160, 160, 3), dtype=np.uint8) * 128

    crops = [crop1, crop2] + [None] * 2
    face_areas = [16000.0, 18000.0, 0.0, 0.0]
    confidences = [0.95, 0.88, 0.0, 0.0]

    q = compute_visual_quality(crops, face_areas=face_areas, detection_confidences=confidences)
    vec = q.to_vector()

    assert len(vec) == 7
    assert vec.dtype == np.float32
    assert np.isclose(q.detection_success_rate, 0.5)
    assert np.isclose(q.failed_frame_fraction, 0.5)
    assert q.blur_variance > 0
    assert q.face_area_mean == 17000.0
    assert q.detection_confidence_mean == pytest.approx((0.95 + 0.88) / 2.0)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_visual_quality_all_failed_crops() -> None:
    """When all face crops are None, returns safe zeroed quality vector with failed fraction 1.0."""
    crops = [None] * 16
    q = compute_visual_quality(crops)
    vec = q.to_vector()

    assert np.isclose(q.detection_success_rate, 0.0)
    assert np.isclose(q.failed_frame_fraction, 1.0)
    assert q.blur_variance == 0.0
    assert q.face_area_mean == 0.0
    assert q.detection_confidence_mean == 0.0
    assert np.all(vec == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_audio_quality_standard_waveform() -> None:
    """2-second 16kHz sine wave produces 5-D audio quality vector."""
    sr = 16_000
    t = np.linspace(0, 2.0, 32_000, endpoint=False, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

    q = compute_audio_quality(waveform, sample_rate=sr, expected_duration_s=2.0)
    vec = q.to_vector()

    assert len(vec) == 5
    assert vec.dtype == np.float32
    assert q.vad_fraction > 0.9
    assert np.isclose(q.rms, 0.5 / np.sqrt(2), atol=1e-2)
    assert q.clipped_sample_ratio == 0.0
    assert q.duration_mismatch == pytest.approx(0.0, abs=1e-5)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_audio_quality_clipped_waveform() -> None:
    """Severely clipped audio reports high clipped_sample_ratio."""
    sr = 16_000
    waveform = np.ones(16_000, dtype=np.float32)  # All samples at 1.0
    q = compute_audio_quality(waveform, sample_rate=sr, expected_duration_s=1.0)

    assert q.clipped_sample_ratio == 1.0
    assert q.rms == 1.0


def test_audio_quality_empty_waveform() -> None:
    """Empty waveform degrades safely to zeros without division by zero."""
    q = compute_audio_quality(np.array([], dtype=np.float32), sample_rate=16_000, expected_duration_s=2.0)
    vec = q.to_vector()

    assert q.vad_fraction == 0.0
    assert q.rms == 0.0
    assert q.clipped_sample_ratio == 0.0
    assert q.duration_mismatch == 2.0
    assert not np.isnan(vec).any()


def test_face_extractor_pickle_and_multiprocessing() -> None:
    """FaceExtractor supports pickle serialization for multi-worker DataLoaders."""
    import pickle
    from src.preprocess.face_extraction import FaceExtractor

    extractor = FaceExtractor(crop_size=160, box_expansion=0.20)
    data = pickle.dumps(extractor)
    restored: FaceExtractor = pickle.loads(data)

    assert restored.crop_size == 160
    assert restored.box_expansion == pytest.approx(0.20)
    assert restored._detector is not None

    # Test dummy frame extraction with restored extractor
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop, bbox, area = restored.extract_face_crop(frame, (10, 10, 50, 50))
    assert crop is not None
    assert crop.shape == (160, 160, 3)
