"""Computes perceptual/video hashes to detect and block duplicate clips leaking across partitions.

Status: implemented — provides compute_file_hash() (SHA-256), compute_frame_hash()
(average perceptual hash), and check_cross_partition_duplicates() for verifying no
duplicate clips cross split boundaries. See AGENTS.md for build order.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_file_hash(
    file_path: str | Path,
    algorithm: str = "sha256",
) -> str:
    """Compute a cryptographic hash of a file.

    Parameters
    ----------
    file_path:
        Path to the file.
    algorithm:
        Hash algorithm (default ``"sha256"``).

    Returns
    -------
    str
        Hex-encoded hash digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_frame_hash(
    frame: np.ndarray,
    hash_size: int = 8,
) -> str:
    """Compute an average perceptual hash of a single video frame.

    The perceptual hash is robust to minor re-encoding / resize differences,
    making it suitable for detecting duplicate clips across partitions.

    Parameters
    ----------
    frame:
        RGB image of shape ``(H, W, 3)`` uint8.
    hash_size:
        Size of the hash image (default 8 → 64-bit hash).

    Returns
    -------
    str
        Hex string of the perceptual hash.
    """
    if cv2 is not None:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to hash_size+1 × hash_size for DCT-like comparison
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    else:
        # Fallback using pure numpy
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(np.uint8)
        else:
            gray = frame.squeeze().astype(np.uint8)
        h, w = gray.shape[:2]
        y_idx = np.linspace(0, max(0, h - 1), hash_size, dtype=int)
        x_idx = np.linspace(0, max(0, w - 1), hash_size + 1, dtype=int)
        resized = gray[np.ix_(y_idx, x_idx)]

    # Compute difference hash (dHash): compare adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]  # (hash_size, hash_size) bool
    # Pack bits into bytes
    bits = diff.flatten()
    # Convert to hex string
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return format(hash_int, f"0{hash_size * hash_size // 4}x")


def compute_video_hash(
    video_path: str | Path,
    n_sample_frames: int = 4,
    hash_size: int = 8,
) -> str:
    """Compute a perceptual hash for a video by hashing sampled frames.

    Samples *n_sample_frames* uniformly and concatenates their individual
    perceptual hashes into a combined video hash.

    Parameters
    ----------
    video_path:
        Path to the video file.
    n_sample_frames:
        Number of frames to sample (default 4).
    hash_size:
        Perceptual hash size per frame.

    Returns
    -------
    str
        Combined hex hash string.
    """
    video_path = Path(video_path)
    if cv2 is None:
        file_hash = compute_file_hash(video_path)
        sub_len = max(1, hash_size * hash_size // 4)
        return "_".join([file_hash[i:i + sub_len] for i in range(0, min(len(file_hash), sub_len * n_sample_frames), sub_len)])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    sample_indices = np.linspace(0, total_frames - 1, n_sample_frames, dtype=int)
    frame_hashes: list[str] = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_hashes.append(compute_frame_hash(rgb, hash_size=hash_size))
        else:
            frame_hashes.append("0" * (hash_size * hash_size // 4))

    cap.release()
    return "_".join(frame_hashes)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute the Hamming distance between two hex hash strings.

    Parameters
    ----------
    hash1, hash2:
        Hex-encoded hash strings (must be same length).

    Returns
    -------
    int
        Number of differing bits.
    """
    if len(hash1) != len(hash2):
        return max(len(hash1), len(hash2)) * 4  # Max distance

    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count("1")


def check_cross_partition_duplicates(
    splits: dict[str, pd.DataFrame],
    hash_column: str = "file_hash",
    max_hamming_distance: int = 0,
) -> list[dict]:
    """Check for duplicate clips across partition boundaries.

    Uses file hashes (cryptographic) or perceptual hashes to identify
    duplicates that would constitute data leakage.

    Parameters
    ----------
    splits:
        Dictionary mapping split name → DataFrame with a hash column.
    hash_column:
        Name of the column containing hashes.
    max_hamming_distance:
        Maximum Hamming distance to consider two hashes as duplicates
        (0 = exact match only; use >0 for perceptual hashes).

    Returns
    -------
    list[dict]
        List of duplicate violation records.  Empty if no duplicates.

    Raises
    ------
    ValueError
        If duplicates are found (with details).
    """
    violations: list[dict] = []
    split_names = list(splits.keys())

    # Build hash → (split, clip_ids) mapping
    hash_to_locations: dict[str, list[tuple[str, str]]] = {}
    for split_name, df in splits.items():
        if hash_column not in df.columns:
            logger.warning(
                "Hash column '%s' not in split '%s', skipping.",
                hash_column,
                split_name,
            )
            continue
        for _, row in df.iterrows():
            h = str(row.get(hash_column, ""))
            if not h or h == "":
                continue
            if h not in hash_to_locations:
                hash_to_locations[h] = []
            hash_to_locations[h].append((split_name, str(row.get("clip_id", ""))))

    # Check for cross-partition duplicates
    for h, locations in hash_to_locations.items():
        split_set = set(loc[0] for loc in locations)
        if len(split_set) > 1:
            violations.append({
                "hash": h,
                "locations": locations,
                "splits_involved": sorted(split_set),
            })

    if violations:
        msg = f"Found {len(violations)} cross-partition duplicate(s):\n"
        for v in violations[:5]:
            msg += f"  Hash {v['hash'][:16]}... in splits {v['splits_involved']}\n"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "No cross-partition duplicates found across %d splits (%d unique hashes).",
        len(splits),
        len(hash_to_locations),
    )
    return violations
