"""Produces the 70/10/20 identity-group-disjoint split and the pre-training cross-manipulation holdout — Section 5.3.

Status: implemented — provides create_identity_disjoint_split() for group-level
splitting and create_cross_manipulation_split() for holding out manipulation
families before training. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_identity_disjoint_split(
    source_table: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    output_dir: str | Path = "manifests/splits",
) -> dict[str, pd.DataFrame]:
    """Create 70/10/20 identity-group-disjoint train/val/test splits.

    Groups all clips of the same ``source_identity`` into the same partition.
    Source-identity groups never cross train/val/test.

    Parameters
    ----------
    source_table:
        The source table DataFrame (from ``manifest_builder.build_source_table``).
    train_ratio:
        Training set ratio (default 0.70).
    val_ratio:
        Validation set ratio (default 0.10).
    test_ratio:
        Test set ratio (default 0.20).
    seed:
        Random seed for reproducible splitting.
    output_dir:
        Directory to save the split manifests.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        the corresponding subset DataFrames.

    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 (within tolerance).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f}"
        )
    required = {"source_identity", "clip_fake"}
    if required - set(source_table.columns):
        raise ValueError(f"Source table is missing split columns: {sorted(required - set(source_table.columns))}")
    if source_table.empty:
        raise ValueError("Cannot create splits from an empty source table")

    # Get unique identities
    identities = source_table["source_identity"].unique()
    n_identities = len(identities)
    logger.info("Splitting %d unique identities into train/val/test", n_identities)

    # Shuffle identities deterministically
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(identities)

    # Compute split boundaries
    if n_identities < 3:
        raise ValueError(f"At least 3 source identities are required for non-empty splits, got {n_identities}")
    n_train = max(1, int(round(n_identities * train_ratio)))
    n_val = max(1, int(round(n_identities * val_ratio)))
    if n_train + n_val >= n_identities:
        n_train = n_identities - 2
        n_val = 1

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])

    # Verify disjointness
    assert train_ids.isdisjoint(val_ids), "Train and val identities overlap"
    assert train_ids.isdisjoint(test_ids), "Train and test identities overlap"
    assert val_ids.isdisjoint(test_ids), "Val and test identities overlap"

    # Build split DataFrames
    splits = {
        "train": source_table[
            source_table["source_identity"].isin(train_ids)
        ].copy(),
        "val": source_table[
            source_table["source_identity"].isin(val_ids)
        ].copy(),
        "test": source_table[
            source_table["source_identity"].isin(test_ids)
        ].copy(),
    }

    # Add split column
    for split_name, df in splits.items():
        df["split"] = split_name

    # Save manifests
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        path = output_dir / f"{split_name}_manifest.csv"
        df.to_csv(path, index=False)
        logger.info(
            "Split '%s': %d clips, %d identities → %s",
            split_name,
            len(df),
            df["source_identity"].nunique(),
            path,
        )

    # Log class balance per split
    for split_name, df in splits.items():
        n_real = int((~df["clip_fake"]).sum())
        n_fake = int(df["clip_fake"].sum())
        logger.info(
            "  %s class balance: real=%d, fake=%d (%.1f%% fake)",
            split_name,
            n_real,
            n_fake,
            100.0 * n_fake / len(df) if len(df) > 0 else 0,
        )

    return splits


def create_cross_manipulation_split(
    source_table: pd.DataFrame,
    held_out_family: str,
    seed: int = 42,
    output_dir: str | Path = "manifests/splits",
) -> dict[str, pd.DataFrame]:
    """Create a cross-manipulation holdout split.

    The held-out manipulation family and its source identities are placed
    entirely in the test set.  Training and validation use only seen families.
    This must be done *before* training (Section 5.3).

    Parameters
    ----------
    source_table:
        The source table DataFrame.
    held_out_family:
        The manipulation family to hold out (e.g., ``"faceswap"``).
    seed:
        Random seed for the seen-family split.
    output_dir:
        Directory to save the cross-manipulation split manifests.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys ``"train"``, ``"val"``, ``"test_seen"``,
        ``"test_held_out"`` mapping to subsets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if held_out_family not in set(source_table["manipulation_family"].astype(str)):
        raise ValueError(f"Held-out family {held_out_family!r} is absent from the source table")

    # Identities represented by the held-out family cannot be used for training
    # or validation, even through another manipulation of the same person.
    held_out_mask = source_table["manipulation_family"] == held_out_family
    held_out_identities = set(
        source_table.loc[held_out_mask, "source_identity"].unique()
    )

    held_out_identity_mask = source_table["source_identity"].isin(held_out_identities)

    # Remaining clips are from seen families/identities
    seen_table = source_table[~held_out_identity_mask].copy()
    held_out_table = source_table[held_out_mask].copy()
    excluded_companions = source_table[held_out_identity_mask & ~held_out_mask]

    logger.info(
        "Cross-manipulation split: held-out family '%s' → %d clips (%d identities), "
        "excluded %d same-identity companion clips, seen → %d clips",
        held_out_family,
        len(held_out_table),
        len(held_out_identities),
        len(excluded_companions),
        len(seen_table),
    )

    # Split the seen data into train/val/test_seen
    if seen_table.empty:
        logger.warning("No seen-family identities remain; cross-manipulation split is not trainable.")
        seen_splits = {name: seen_table.copy() for name in ("train", "val", "test")}
    else:
        seen_splits = create_identity_disjoint_split(
            seen_table,
            train_ratio=0.75,
            val_ratio=0.10,
            test_ratio=0.15,
            seed=seed,
            output_dir=output_dir,
        )

    # Add the held-out test set
    held_out_table["split"] = "test_held_out"
    seen_splits["test_seen"] = seen_splits.pop("test")
    seen_splits["test_held_out"] = held_out_table
    seen_splits["test_seen"].to_csv(Path(output_dir) / "test_seen_manifest.csv", index=False)

    # Save held-out manifest
    output_path = output_dir / "test_held_out_manifest.csv"
    held_out_table.to_csv(output_path, index=False)
    logger.info(
        "Held-out test set: %d clips → %s",
        len(held_out_table),
        output_path,
    )

    return seen_splits


def verify_split_integrity(
    splits: dict[str, pd.DataFrame],
) -> bool:
    """Verify that no source identity appears in more than one partition.

    Parameters
    ----------
    splits:
        Dictionary of split name → DataFrame.

    Returns
    -------
    bool
        ``True`` if all splits are identity-disjoint.

    Raises
    ------
    ValueError
        If any identity is found in multiple partitions.
    """
    identity_to_splits: dict[str, list[str]] = {}

    for split_name, df in splits.items():
        for identity in df["source_identity"].unique():
            identity = str(identity)
            if identity not in identity_to_splits:
                identity_to_splits[identity] = []
            identity_to_splits[identity].append(split_name)

    violations = {
        identity: split_list
        for identity, split_list in identity_to_splits.items()
        if len(split_list) > 1
    }

    if violations:
        msg = "Identity-split violations:\n"
        for identity, split_list in list(violations.items())[:10]:
            msg += f"  {identity} → {split_list}\n"
        raise ValueError(msg)

    logger.info(
        "Split integrity verified: %d identities across %d partitions, no overlaps.",
        len(identity_to_splits),
        len(splits),
    )
    return True
