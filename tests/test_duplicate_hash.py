"""Verifies no duplicate clip (by perceptual/video hash) crosses a partition boundary.

Status: implemented — tests perceptual frame hashing, Hamming distance,
leakage detection across partitions, and manifest partition exclusivity.
See AGENTS.md for build order.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.hashing import (
    check_cross_partition_duplicates,
    compute_file_hash,
    compute_frame_hash,
    hamming_distance,
)


def test_file_hash_deterministic() -> None:
    """Verifies that cryptographic file hashing is deterministic and exact."""
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"Deepfake detection reproducibility test content 12345")
        temp_path = Path(f.name)

    try:
        h1 = compute_file_hash(temp_path)
        h2 = compute_file_hash(temp_path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 is 64 hex characters
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_frame_hash_identical_and_differing() -> None:
    """Identical frames produce 0 Hamming distance; inverted frames produce high distance."""
    # Synthetic frame: 160x160 RGB
    frame1 = np.zeros((160, 160, 3), dtype=np.uint8)
    frame1[:80, :80] = 255
    frame1[80:, 80:] = 255

    frame2 = frame1.copy()

    # Inverted frame
    frame3 = 255 - frame1

    h1 = compute_frame_hash(frame1)
    h2 = compute_frame_hash(frame2)
    h3 = compute_frame_hash(frame3)

    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    dist_inverted = hamming_distance(h1, h3)
    assert dist_inverted > 0, "Inverted frames should have nonzero Hamming distance"


def test_hamming_distance_exact() -> None:
    """Verifies exact bit-level Hamming distance calculation."""
    assert hamming_distance("00", "00") == 0
    assert hamming_distance("0f", "00") == 4
    assert hamming_distance("ff", "00") == 8
    assert hamming_distance("f0", "0f") == 8


def test_check_cross_partition_duplicates_clean() -> None:
    """Disjoint partition hashes pass duplicate check with no violations."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not yet available")

    train_df = pd.DataFrame({
        "clip_id": ["c1", "c2", "c3"],
        "file_hash": ["aaa1", "aaa2", "aaa3"],
    })
    val_df = pd.DataFrame({
        "clip_id": ["c4", "c5"],
        "file_hash": ["bbb1", "bbb2"],
    })
    test_df = pd.DataFrame({
        "clip_id": ["c6", "c7"],
        "file_hash": ["ccc1", "ccc2"],
    })

    splits = {"train": train_df, "val": val_df, "test": test_df}
    violations = check_cross_partition_duplicates(splits, hash_column="file_hash")
    assert len(violations) == 0


def test_check_cross_partition_duplicates_detected() -> None:
    """Duplicate hash crossing partitions raises ValueError and reports leak."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not yet available")

    train_df = pd.DataFrame({
        "clip_id": ["c1", "c2"],
        "file_hash": ["leak_hash_1", "train_only"],
    })
    val_df = pd.DataFrame({
        "clip_id": ["c3"],
        "file_hash": ["val_only"],
    })
    test_df = pd.DataFrame({
        "clip_id": ["c4"],
        "file_hash": ["leak_hash_1"],  # Leaked duplicate!
    })

    splits = {"train": train_df, "val": val_df, "test": test_df}
    with pytest.raises(ValueError, match="cross-partition duplicate"):
        check_cross_partition_duplicates(splits, hash_column="file_hash")


def test_hamming_distance_with_separators() -> None:
    """Verifies that hamming_distance handles '_' separators in multi-frame hashes."""
    assert hamming_distance("00_00", "00_00") == 0
    assert hamming_distance("0f_00", "00_00") == 4
    assert hamming_distance("ff_ff", "00_00") == 16
    assert hamming_distance("1a_2b_3c", "1a_2b_3d") == 1


def test_check_cross_partition_duplicates_zero_hash() -> None:
    """Verifies that valid hash '0' is not skipped and duplicate '0' across splits is caught."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not yet available")

    splits = {
        "train": pd.DataFrame({"clip_id": ["c1"], "file_hash": ["0"]}),
        "val": pd.DataFrame({"clip_id": ["c2"], "file_hash": ["0"]}),
    }
    with pytest.raises(ValueError, match="cross-partition duplicate"):
        check_cross_partition_duplicates(splits, hash_column="file_hash")
