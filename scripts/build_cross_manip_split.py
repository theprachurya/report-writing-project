"""Build a pre-training identity-disjoint cross-manipulation split from the source table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_yaml
from src.utils.splitting import create_cross_manipulation_split


def parse_args() -> argparse.Namespace:
    """Parse source manifest, family choice, ratios, and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table", default="manifests/source_table.csv")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--family", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    """Write train/validation and held-out test manifests before model training."""
    args = parse_args()
    source_path = Path(args.source_table)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Source table not found: {source_path}. Run the data audit first.")

    eval_config = load_yaml(args.eval_config)
    data_config = load_yaml(args.data_config)
    family = args.family or eval_config.get("cross_manipulation", {}).get("held_out_family")
    if not family:
        raise ValueError("Choose a held-out manipulation family in eval_config.yaml or --family")
    split_cfg = data_config.get("cross_manipulation_split", {})
    seed = args.seed if args.seed is not None else int(data_config.get("split", {}).get("seed", 42))
    output_dir = args.output_dir or f"manifests/splits/cross_manipulation_{family}"
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    source_table = pd.read_csv(source_path)
    splits = create_cross_manipulation_split(
        source_table,
        held_out_family=str(family),
        seed=seed,
        output_dir=output_path,
        train_ratio=float(split_cfg.get("train", 0.75)),
        val_ratio=float(split_cfg.get("val", 0.10)),
        test_ratio=float(split_cfg.get("test_seen", 0.15)),
    )
    metadata = {
        "held_out_family": str(family),
        "seed": seed,
        "source_table": str(source_path),
        "source_table_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "seen_identity_ratios": {
            "train": float(split_cfg.get("train", 0.75)),
            "val": float(split_cfg.get("val", 0.10)),
            "test_seen": float(split_cfg.get("test_seen", 0.15)),
        },
        "splits": {},
    }
    for name, frame in splits.items():
        labels = Counter(str(value).lower() for value in frame.get("clip_fake", []))
        metadata["splits"][name] = {
            "clips": int(len(frame)),
            "source_identities": int(frame["source_identity"].nunique()) if "source_identity" in frame else 0,
            "clip_fake_counts": dict(labels),
            "manipulation_families": dict(Counter(frame["manipulation_family"].astype(str)))
            if "manipulation_family" in frame else {},
        }
    (output_path / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Cross-manipulation manifests written to {output_path}")
    print(json.dumps(metadata["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
