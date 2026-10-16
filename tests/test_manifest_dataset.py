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


def test_manifest_dataset_cache_key_includes_dataset(tmp_path) -> None:
    """Verifies cache key includes dataset name to prevent cross-dataset collisions."""
    clip_id = "shared-id"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Pre-save cached files using the dataset:clip_id hashing convention
    key_fakeav = hashlib.sha256(f"fakeavceleb:{clip_id}".encode("utf-8")).hexdigest() + ".npz"
    key_dfdc = hashlib.sha256(f"dfdc:{clip_id}".encode("utf-8")).hexdigest() + ".npz"
    assert key_fakeav != key_dfdc

    np.savez_compressed(
        cache_dir / key_fakeav,
        frames=np.full((16, 160, 160, 3), 100, dtype=np.uint8),
        mel_spec=np.zeros((80, 201), dtype=np.float32),
        quality_v=np.zeros(7, dtype=np.float32),
        quality_a=np.zeros(5, dtype=np.float32),
        visual_missing=np.float32(0),
        audio_missing=np.float32(0),
    )
    np.savez_compressed(
        cache_dir / key_dfdc,
        frames=np.full((16, 160, 160, 3), 200, dtype=np.uint8),
        mel_spec=np.zeros((80, 201), dtype=np.float32),
        quality_v=np.zeros(7, dtype=np.float32),
        quality_a=np.zeros(5, dtype=np.float32),
        visual_missing=np.float32(0),
        audio_missing=np.float32(0),
    )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {
            "clip_id": clip_id,
            "dataset": "fakeavceleb",
            "file_path": "fakeav.mp4",
            "source_identity": "person-1",
            "clip_fake": 0,
        },
        {
            "clip_id": clip_id,
            "dataset": "dfdc",
            "file_path": "dfdc.mp4",
            "source_identity": "person-2",
            "clip_fake": 1,
        },
    ]).to_csv(manifest_path, index=False)

    dataset = ManifestAVDataset(
        manifest_path,
        {"fakeavceleb": tmp_path, "dfdc": tmp_path},
        cache_dir,
        load_data_config(),
    )
    item0 = dataset[0]
    item1 = dataset[1]
    assert item0["target"].item() == 0.0
    assert item1["target"].item() == 1.0


def test_manifest_dataset_re_derives_validity_mask_on_frame_drop(tmp_path) -> None:
    """Verifies face_valid_mask and visual_missing are re-derived after video frame drops."""
    clip_id = "test-drop"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = hashlib.sha256(f"fakeavceleb:{clip_id}".encode("utf-8")).hexdigest() + ".npz"
    np.savez_compressed(
        cache_dir / key,
        frames=np.full((16, 160, 160, 3), 128, dtype=np.uint8),
        face_valid_mask=np.ones(16, dtype=bool),
        face_areas=np.full(16, 1000.0, dtype=np.float32),
        detection_confidences=np.ones(16, dtype=np.float32),
        mel_spec=np.zeros((80, 201), dtype=np.float32),
        quality_v=np.zeros(7, dtype=np.float32),
        quality_a=np.zeros(5, dtype=np.float32),
        visual_missing=np.float32(0),
        audio_missing=np.float32(0),
    )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "clip_id": clip_id,
        "dataset": "fakeavceleb",
        "file_path": "drop.mp4",
        "source_identity": "person-1",
        "clip_fake": 0,
    }]).to_csv(manifest_path, index=False)

    aug_config = {
        "video": {
            "enabled": True,
            "resize_prob": 0.0,
            "blur_prob": 0.0,
            "frame_drop_prob": 1.0,
            "reencode_prob": 0.0,
            "noise_prob": 0.0,
        },
        "audio": {"enabled": False},
    }

    dataset = ManifestAVDataset(
        manifest_path,
        {"fakeavceleb": tmp_path},
        cache_dir,
        load_data_config(),
        training=True,
        augmentation_config=aug_config,
    )
    item = dataset[0]
    assert item["quality_v"].shape == (7,)
