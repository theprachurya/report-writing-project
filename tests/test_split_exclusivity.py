"""Verifies no source identity appears in more than one of train/val/test.

Status: implemented — tests that identity-group-disjoint splitting enforces strict
partition isolation (train, val, test identity sets are completely disjoint),
validates ratio constraints, and checks cross-manipulation holdouts.
See AGENTS.md for build order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.splitting import (
    create_cross_manipulation_split,
    create_identity_disjoint_split,
)


def _make_synthetic_source_table():
    """Helper creating synthetic source table DataFrame with multiple identities."""
    try:
        import pandas as pd
    except ImportError:
        return None

    identities = [f"person_{i:03d}" for i in range(25)]
    records = []
    for ident in identities:
        # 4 clips per identity: 2 real, 2 fake
        for clip_num in range(4):
            records.append({
                "clip_id": f"{ident}_clip_{clip_num}",
                "source_identity": ident,
                "clip_fake": (clip_num >= 2),
                "manipulation_family": "real" if clip_num < 2 else ("faceswap" if clip_num == 2 else "wav2lip"),
                "file_hash": f"hash_{ident}_{clip_num}",
                "video_rel_path": f"videos/{ident}_{clip_num}.mp4",
                "audio_rel_path": f"audios/{ident}_{clip_num}.wav",
                "dataset_name": "fakeavceleb",
            })
    return pd.DataFrame(records)


def test_identity_disjoint_split_exclusivity(tmp_path: Path) -> None:
    """Verifies that no source identity appears in more than one of train/val/test."""
    df = _make_synthetic_source_table()
    if df is None:
        pytest.skip("pandas not yet available")

    splits = create_identity_disjoint_split(
        source_table=df,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=42,
        output_dir=tmp_path / "splits",
    )

    train_ids = set(splits["train"]["source_identity"].unique())
    val_ids = set(splits["val"]["source_identity"].unique())
    test_ids = set(splits["test"]["source_identity"].unique())

    all_ids = set(df["source_identity"].unique())

    # 1. Pairwise disjointness
    assert train_ids.isdisjoint(val_ids), f"Train and val share identities: {train_ids & val_ids}"
    assert train_ids.isdisjoint(test_ids), f"Train and test share identities: {train_ids & test_ids}"
    assert val_ids.isdisjoint(test_ids), f"Val and test share identities: {val_ids & test_ids}"

    # 2. Complete partition coverage
    assert train_ids | val_ids | test_ids == all_ids, "Missing identities from split union"

    # 3. Non-empty splits
    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert len(test_ids) > 0


def test_split_ratios_validation() -> None:
    """Ratios that don't sum to 1.0 raise ValueError."""
    df = _make_synthetic_source_table()
    if df is None:
        pytest.skip("pandas not yet available")

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        create_identity_disjoint_split(
            source_table=df,
            train_ratio=0.80,
            val_ratio=0.15,
            test_ratio=0.20,  # Sum = 1.15
        )


def test_cross_manipulation_holdout_exclusivity(tmp_path: Path) -> None:
    """Held-out manipulation family never leaks into training or validation splits."""
    df = _make_synthetic_source_table()
    if df is None:
        pytest.skip("pandas not yet available")

    splits = create_cross_manipulation_split(
        source_table=df,
        held_out_family="faceswap",
        seed=42,
        output_dir=tmp_path / "cross_manip",
    )

    # faceswap must NEVER appear in train or val
    train_fams = set(splits["train"]["manipulation_family"].unique())
    val_fams = set(splits["val"]["manipulation_family"].unique())
    test_held_fams = set(splits["test_held_out"]["manipulation_family"].unique())

    assert "faceswap" not in train_fams, "Held-out family found in train set!"
    assert "faceswap" not in val_fams, "Held-out family found in val set!"
    assert test_held_fams == {"faceswap"}, "test_held_out should contain only the held-out family"

    # Held-out family's identities must NEVER appear in train or val
    held_out_ids = set(splits["test_held_out"]["source_identity"].unique())
    train_ids = set(splits["train"]["source_identity"].unique())
    val_ids = set(splits["val"]["source_identity"].unique())
    assert held_out_ids.isdisjoint(train_ids), "Held-out identities leaked into train set!"
    assert held_out_ids.isdisjoint(val_ids), "Held-out identities leaked into val set!"


def test_split_with_string_clip_fake(tmp_path: Path) -> None:
    """Verifies that create_identity_disjoint_split handles string/int clip_fake safely."""
    df = _make_synthetic_source_table()
    if df is None:
        pytest.skip("pandas not yet available")
    df["clip_fake"] = df["clip_fake"].astype(str)  # 'True' / 'False' strings
    splits = create_identity_disjoint_split(
        source_table=df,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=42,
        output_dir=tmp_path / "splits_str",
    )
    assert len(splits["train"]) > 0
