"""Precompute and cache extracted features for split manifests ahead of training.

Decodes video and audio windows, extracts face crops, computes log-Mel spectrograms
and quality features, and writes .npz cache files so that training epochs do not
repeat expensive media decoding on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest_dataset import dataset_roots_from_config
from src.preprocess.feature_cache import preprocess_manifest_row
from src.utils.config_loader import load_data_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("precompute_cache")


def _process_single_row(
    row_dict: dict[str, Any],
    dataset_roots: dict[str, Path],
    cache_dir: Path,
    data_config: Any,
) -> tuple[str, bool, str | None]:
    """Process a single manifest row and return status."""
    clip_id = str(row_dict["clip_id"])
    cache_name = hashlib.sha256(clip_id.encode("utf-8")).hexdigest() + ".npz"
    cache_path = cache_dir / cache_name

    if cache_path.is_file():
        return clip_id, True, None

    dataset_name = str(row_dict.get("dataset", "fakeavceleb")).lower()
    if dataset_name not in dataset_roots:
        return clip_id, False, f"Missing root for {dataset_name}"

    try:
        preprocess_manifest_row(
            row=row_dict,
            dataset_root=dataset_roots[dataset_name],
            output_path=cache_path,
            data_config=data_config,
        )
        return clip_id, False, None
    except Exception as exc:
        return clip_id, False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute and cache features from split manifests."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/splits/val_manifest.csv",
        help="Path to manifest CSV (default: manifests/splits/val_manifest.csv).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration YAML (default: configs/data_config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cache/features",
        help="Output cache directory (default: data/cache/features).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum clips to process (optional).",
    )
    args = parser.parse_args()
    setup_logging(level=logging.INFO)

    cfg = load_data_config(config_path=args.config)
    dataset_roots = dataset_roots_from_config(cfg)
    if not dataset_roots:
        logger.error("No valid dataset roots configured.")
        return 1

    # Convert config to dict for safe multiprocessing pickling
    import dataclasses
    cfg_payload = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    df = pd.read_csv(manifest_path)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    cache_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict("records")
    logger.info("Precomputing features for %d clips using %d workers → %s", len(rows), args.workers, cache_dir)

    already_cached = 0
    newly_cached = 0
    failed = 0

    if args.workers <= 1:
        for idx, row in enumerate(rows, 1):
            clip_id, was_cached, error = _process_single_row(row, dataset_roots, cache_dir, cfg_payload)
            if was_cached:
                already_cached += 1
            elif error:
                failed += 1
                logger.warning("[%d/%d] Failed %s: %s", idx, len(rows), clip_id, error)
            else:
                newly_cached += 1
            if idx % 50 == 0 or idx == len(rows):
                logger.info("[%d/%d] Cached: %d new, %d already present, %d failed", idx, len(rows), newly_cached, already_cached, failed)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_single_row, row, dataset_roots, cache_dir, cfg_payload): row["clip_id"]
                for row in rows
            }
            total = len(futures)
            for idx, fut in enumerate(as_completed(futures), 1):
                clip_id, was_cached, error = fut.result()
                if was_cached:
                    already_cached += 1
                elif error:
                    failed += 1
                    logger.warning("[%d/%d] Failed %s: %s", idx, total, clip_id, error)
                else:
                    newly_cached += 1
                if idx % 50 == 0 or idx == total:
                    logger.info("[%d/%d] Cached: %d new, %d already present, %d failed", idx, total, newly_cached, already_cached, failed)

    logger.info(
        "Finished precomputation: %d newly cached, %d previously cached, %d failed.",
        newly_cached,
        already_cached,
        failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
