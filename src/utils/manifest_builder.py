"""Builds manifests/source_table.csv from raw dataset metadata (identity, manipulation family, audio condition) before any augmentation.

Status: implemented — provides build_source_table() for scanning dataset directories
and assembling the pre-augmentation source table with identity grouping, manipulation
family, and audio condition metadata. See AGENTS.md for build order.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column schema for source_table.csv ──────────────────────────────────────
# Per Section 5.3: source identity, source video, dataset, manipulation family,
# audio condition, and clip ID.
SOURCE_TABLE_COLUMNS = [
    "clip_id",              # Unique identifier for each clip
    "dataset",              # "fakeavceleb" | "dfdc"
    "source_identity",      # Identity / subject ID (for group splitting)
    "source_video",         # Original source video filename
    "manipulation_family",  # e.g., "faceswap", "reenactment", "tts", "vc", "real"
    "video_manipulated",    # bool: was the video track manipulated?
    "audio_manipulated",    # bool: was the audio track manipulated?
    "clip_fake",            # bool: derived label (documented rule)
    "audio_condition",      # e.g., "original", "tts", "voice_conversion", "unknown"
    "file_path",            # Relative path to the clip file
    "duration_s",           # Clip duration in seconds (if known)
    "file_hash",            # SHA-256 hash for deduplication
]


def _compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute a cryptographic hash of a file for deduplication.

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
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_fakeavceleb_metadata(
    root: Path,
) -> list[dict[str, Any]]:
    """Parse FakeAVCeleb dataset directory structure into source-table rows.

    Expected layout (approximate):
    ``<root>/RealVideo-RealAudio/<identity>/*.mp4``
    ``<root>/FakeVideo-RealAudio/<method>/<identity>/*.mp4``
    ``<root>/FakeVideo-FakeAudio/<method>/<identity>/*.mp4``
    ``<root>/RealVideo-FakeAudio/<identity>/*.mp4``

    Parameters
    ----------
    root:
        Dataset root directory.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the source table.
    """
    rows: list[dict[str, Any]] = []
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name.lower()

        # Determine manipulation status from category name
        if "realvideo" in category and "realaudio" in category:
            video_manip = False
            audio_manip = False
            manip_family = "real"
            audio_cond = "original"
        elif "fakevideo" in category and "realaudio" in category:
            video_manip = True
            audio_manip = False
            manip_family = "unknown"  # Will be refined by sub-directory
            audio_cond = "original"
        elif "fakevideo" in category and "fakeaudio" in category:
            video_manip = True
            audio_manip = True
            manip_family = "unknown"
            audio_cond = "unknown"
        elif "realvideo" in category and "fakeaudio" in category:
            video_manip = False
            audio_manip = True
            manip_family = "real"
            audio_cond = "tts"
        else:
            logger.warning("Unrecognized category: %s", category)
            continue

        # Walk the directory tree
        for clip_path in sorted(category_dir.rglob("*")):
            if not clip_path.is_file():
                continue
            if clip_path.suffix.lower() not in video_extensions:
                continue

            # Extract identity from directory structure
            relative = clip_path.relative_to(category_dir)
            parts = relative.parts

            if (video_manip or audio_manip) and len(parts) >= 3:
                # Common FakeAVCeleb form: method/identity/clip.ext.
                method = parts[0].lower()
                identity = parts[-2]
                if video_manip:
                    clip_family = method
                else:
                    clip_family = f"audio_{method}"
                if audio_manip:
                    audio_cond = "tts" if "tts" in method else "voice_conversion"
            elif video_manip or audio_manip:
                # Some releases omit one directory level; preserve a stable
                # identity key and mark the family conservatively.
                method = parts[0].lower() if parts else "unknown"
                identity = parts[-2] if len(parts) >= 2 else "unknown"
                clip_family = method if video_manip else f"audio_{audio_cond}"
            else:
                # Real: identity/file.mp4
                identity = parts[0] if parts else "unknown"
                clip_family = manip_family

            relative_path = str(clip_path.relative_to(root))
            stable_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
            clip_id = f"fakeavceleb_{stable_id}"
            rows.append({
                "clip_id": clip_id,
                "dataset": "fakeavceleb",
                "source_identity": str(identity),
                "source_video": clip_path.stem,
                "manipulation_family": clip_family,
                "video_manipulated": video_manip,
                "audio_manipulated": audio_manip,
                "clip_fake": video_manip or audio_manip,
                "audio_condition": audio_cond,
                "file_path": relative_path,
                "duration_s": None,
                "file_hash": "",
            })
    return rows


def _parse_dfdc_metadata(
    root: Path,
    metadata_file: str = "metadata.json",
) -> list[dict[str, Any]]:
    """Parse DFDC dataset structure into source-table rows.

    DFDC is structured as numbered directories (dfdc_train_part_XX) each
    containing an ``metadata.json`` that maps filename → label and original.

    Parameters
    ----------
    root:
        Dataset root directory.
    metadata_file:
        Name of the per-part metadata JSON file.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the source table.
    """
    import json

    rows: list[dict[str, Any]] = []
    video_extensions = {".mp4", ".avi"}

    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir():
            continue

        meta_path = part_dir / metadata_file
        metadata: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        for clip_path in sorted(part_dir.glob("*")):
            if not clip_path.is_file():
                continue
            if clip_path.suffix.lower() not in video_extensions:
                continue

            fname = clip_path.name
            meta = metadata.get(fname, {})
            label = meta.get("label", "UNKNOWN").upper()
            original = meta.get("original", fname)

            is_fake = label == "FAKE"
            # DFDC is fundamentally a face-swap benchmark (Section 5.1)
            # Do NOT claim audio is synthetic unless audited
            identity = original.replace(".mp4", "")

            rows.append({
                "clip_id": f"dfdc_{hashlib.sha256(str(clip_path.relative_to(root)).encode('utf-8')).hexdigest()[:16]}",
                "dataset": "dfdc",
                "source_identity": identity,
                "source_video": original,
                "manipulation_family": "faceswap" if is_fake else "real",
                "video_manipulated": is_fake,
                "audio_manipulated": False,  # Conservative: DFDC audio not claimed synthetic
                "clip_fake": is_fake,
                "audio_condition": "original",
                "file_path": str(clip_path.relative_to(root)),
                "duration_s": None,
                "file_hash": "",
            })

    return rows


def build_source_table(
    dataset_roots: dict[str, str | Path],
    output_path: str | Path = "manifests/source_table.csv",
    compute_hashes: bool = True,
) -> pd.DataFrame:
    """Build the source table from raw dataset directories.

    This must be called *before* any augmentation, splitting, or training
    (Section 5.3).  The output is a versioned CSV manifest.

    Parameters
    ----------
    dataset_roots:
        Mapping of dataset name → root directory path.
        Supported keys: ``"fakeavceleb"``, ``"dfdc"``.
    output_path:
        Where to save the source table CSV.
    compute_hashes:
        Whether to compute SHA-256 file hashes (slow but recommended).

    Returns
    -------
    pd.DataFrame
        The assembled source table.
    """
    all_rows: list[dict[str, Any]] = []

    for name, root_str in dataset_roots.items():
        root = Path(root_str)
        if not root.exists():
            logger.warning("Dataset root not found, skipping: %s (%s)", name, root)
            continue

        name_lower = name.lower()
        if name_lower == "fakeavceleb":
            rows = _parse_fakeavceleb_metadata(root)
        elif name_lower == "dfdc":
            rows = _parse_dfdc_metadata(root)
        else:
            logger.warning("Unknown dataset '%s', skipping.", name)
            continue

        logger.info("Parsed %d clips from %s", len(rows), name)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=SOURCE_TABLE_COLUMNS)

    # Compute file hashes for deduplication
    if compute_hashes:
        for name, root_str in dataset_roots.items():
            root = Path(root_str)
            if not root.exists():
                continue
            mask = df["dataset"] == name.lower()
            for idx in df.index[mask]:
                file_path = root / df.at[idx, "file_path"]
                if file_path.exists():
                    df.at[idx, "file_hash"] = _compute_file_hash(file_path)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Source table saved: %d clips → %s (real=%d, fake=%d)",
        len(df),
        output_path,
        int((~df["clip_fake"]).sum()),
        int(df["clip_fake"].sum()),
    )

    return df


def validate_source_table(source_table: pd.DataFrame) -> None:
    """Validate the required source metadata and reject ambiguous training rows."""
    required = set(SOURCE_TABLE_COLUMNS)
    missing = required - set(source_table.columns)
    if missing:
        raise ValueError(f"Source table is missing columns: {sorted(missing)}")
    if source_table.empty:
        raise ValueError("Source table contains no clips")
    if source_table["clip_id"].isna().any() or source_table["clip_id"].duplicated().any():
        raise ValueError("clip_id values must be present and unique")
    if source_table["source_identity"].isna().any() or (source_table["source_identity"].astype(str).str.len() == 0).any():
        raise ValueError("Every clip must have a non-empty source_identity")
    labels = pd.to_numeric(source_table["clip_fake"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("clip_fake must contain only binary labels 0/1")
    empty_paths = source_table["file_path"].isna() | (source_table["file_path"].astype(str).str.len() == 0)
    if empty_paths.any():
        raise ValueError("Every clip must have a non-empty file_path")
