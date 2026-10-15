"""Verifies cached manifest features load with stable tensor shapes and labels."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.data.manifest_dataset import ManifestAVDataset
from src.utils.config_loader import load_data_config


def test_manifest_dataset_loads_cached_features(tmp_path) -> None:
    """Reads cached fixture features without opening any licensed media files."""
    clip_id = "fixture-clip"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = hashlib.sha256(clip_id.encode("utf-8")).hexdigest() + ".npz"
    np.savez_compressed(
        cache_dir / key,
        frames=np.full((16, 160, 160, 3), 128, dtype=np.uint8),
        mel_spec=np.zeros((80, 201), dtype=np.float32),
        quality_v=np.zeros(7, dtype=np.float32),
        quality_a=np.zeros(5, dtype=np.float32),
        visual_missing=np.float32(0),
        audio_missing=np.float32(0),
    )
    manifest_path = tmp_path / "train_manifest.csv"
    pd.DataFrame([{
        "clip_id": clip_id,
        "dataset": "fakeavceleb",
        "file_path": "not-opened.mp4",
        "source_identity": "fixture-person",
        "clip_fake": 1,
    }]).to_csv(manifest_path, index=False)

    dataset = ManifestAVDataset(
        manifest_path,
        {"fakeavceleb": tmp_path},
        cache_dir,
        load_data_config(),
    )
    item = dataset[0]

    assert item["frames"].shape == (16, 3, 160, 160)
    assert item["mel_spec"].shape == (1, 80, 201)
    assert item["quality_v"].shape == (7,)
    assert item["quality_a"].shape == (5,)
    assert item["target"].item() == 1.0
    assert item["visual_missing"].item() == 0.0
    assert np.isfinite(item["frames"].numpy()).all()
