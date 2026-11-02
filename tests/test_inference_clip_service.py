"""Unit tests for single-clip inference service (src/inference/clip_service.py)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch

from src.inference.clip_service import ClipPredictionService, InferenceInputError


CHECKPOINT_PATH = Path("artifacts/checkpoints/reliability_fusion/best_model.pt")


def test_service_initialization_missing_checkpoint() -> None:
    """Raises FileNotFoundError if checkpoint does not exist."""
    service = ClipPredictionService("nonexistent/checkpoint.pt")
    with pytest.raises(FileNotFoundError):
        service.load_model()


def test_service_status_fields() -> None:
    """Status dictionary contains all required health and configuration keys."""
    service = ClipPredictionService(CHECKPOINT_PATH)
    status = service.status()
    assert isinstance(status, dict)
    assert "ready" in status
    assert "checkpoint" in status
    assert "threshold_ready" in status
    assert "threshold_source" in status
    assert "device" in status
    assert "window_duration_s" in status
    assert "max_upload_mb" in status
    assert "max_clip_duration_s" in status
    assert status["checkpoint"] == "best_model.pt"


def test_service_load_model() -> None:
    """Model and checkpoint load cleanly, enabling ready status."""
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Checkpoint not available at {CHECKPOINT_PATH}")

    service = ClipPredictionService(CHECKPOINT_PATH)
    assert not service.status()["ready"]
    service.load_model()
    assert service.status()["ready"]
    assert service.model is not None


def test_input_validation_disallowed_extension() -> None:
    """Rejects files with disallowed extensions."""
    service = ClipPredictionService(CHECKPOINT_PATH)
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        with pytest.raises(InferenceInputError, match="Unsupported video format"):
            service.predict(temp_file.name)


def test_input_validation_nonexistent_file() -> None:
    """Rejects nonexistent files."""
    service = ClipPredictionService(CHECKPOINT_PATH)
    with pytest.raises(InferenceInputError, match="could not be found"):
        service.predict("nonexistent_video.mp4")


def test_threshold_cache_generation_and_load() -> None:
    """Validates threshold cache file format and loading."""
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Checkpoint not available at {CHECKPOINT_PATH}")

    service = ClipPredictionService(CHECKPOINT_PATH)
    service.load_model()
    val_manifest = service.manifest_dir / "val_manifest.csv"
    if val_manifest.is_file():
        cache_path = service._threshold_cache_path(val_manifest)
        if cache_path.is_file():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            assert "threshold" in data
            assert "method" in data
            assert isinstance(data["threshold"], float)
            assert 0.0 <= data["threshold"] <= 1.0
