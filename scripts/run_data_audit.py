"""CLI entry point for staged-training step 1: decode + preprocess + build/validate manifests on a small clip sample.

Status: implemented — provides end-to-end data audit, preprocessing pipeline verification,
source table assembly, and split manifest validation per Section 6 and Section 14 (Week 2).
See AGENTS.md for build order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path so 'src' is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decode.audio_decoder import decode_audio_window
from src.decode.video_decoder import decode_video_window
from src.preprocess.alignment import TemporalAligner
from src.preprocess.audio_features import compute_log_mel_spectrogram
from src.utils.config_loader import load_data_config
from src.utils.hashing import check_cross_partition_duplicates, compute_file_hash
from src.utils.logging_utils import setup_logging
from src.utils.manifest_builder import build_source_table, validate_source_table
from src.utils.seed_utils import set_seed
from src.utils.splitting import create_identity_disjoint_split
from src.utils.splitting import verify_split_integrity

logger = logging.getLogger("run_data_audit")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run deepfake detection data audit and manifest validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration YAML file (default: configs/data_config.yaml)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override path to raw dataset root directory.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=200,
        help="Maximum clips to inspect during the audit (default: 200 per Week 2 milestone).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="manifests",
        help="Output directory for manifests and audit report (default: manifests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="Skip SHA-256 hashing during source-manifest creation (faster; weaker duplicate audit).",
    )
    return parser.parse_args()


def audit_clip(
    video_path: Path,
    audio_path: Path | None,
    cfg: Any,
) -> dict[str, Any]:
    """Inspect and validate a single audio-visual clip.

    Verifies 25 fps decode contract, 2s/16-frame windows, log-Mel spectrogram,
    and temporal alignment.
    """
    record: dict[str, Any] = {
        "video_path": str(video_path),
        "video_valid": False,
        "audio_valid": False,
        "n_frames": 0,
        "has_clipping": False,
        "hash": "",
    }

    try:
        record["hash"] = compute_file_hash(video_path)
    except Exception as exc:
        logger.debug("Failed to hash file %s: %s", video_path, exc)

    # 1. Video decode audit
    try:
        vid_res = decode_video_window(
            video_path,
            start_s=0.0,
            window_duration_s=cfg.decode.window_duration_s,
            frames_per_window=cfg.decode.frames_per_window,
            target_fps=cfg.decode.target_fps,
        )
        record["n_frames"] = len(vid_res.frames)
        record["video_valid"] = (len(vid_res.frames) == cfg.decode.frames_per_window)
    except Exception as exc:
        logger.debug("Video decode failed for %s: %s", video_path, exc)

    # 2. Audio decode & spectrogram audit
    target_audio = audio_path if audio_path and audio_path.exists() else video_path
    try:
        aud_res = decode_audio_window(
            target_audio,
            start_s=0.0,
            window_duration_s=cfg.decode.window_duration_s,
            target_sr=cfg.audio.sample_rate,
        )
        record["audio_valid"] = len(aud_res.waveform) > 0
        record["has_clipping"] = aud_res.clipped_sample_ratio > 0.01

        # Mel spectrogram computation
        mel_res = compute_log_mel_spectrogram(
            waveform=aud_res.waveform,
            sample_rate=cfg.audio.sample_rate,
            n_mels=cfg.mel.n_mels,
            window_length_ms=cfg.mel.window_length_ms,
            hop_length_ms=cfg.mel.hop_length_ms,
            f_min=cfg.mel.f_min,
            f_max=cfg.mel.f_max,
        )
        record["mel_valid"] = (mel_res.spectrogram.shape[0] == cfg.mel.n_mels)
    except Exception as exc:
        logger.debug("Audio decode/mel failed for %s: %s", target_audio, exc)
        record["mel_valid"] = False

    # 3. Temporal alignment check
    try:
        aligner = TemporalAligner(
            n_bins=cfg.alignment.temporal_bins,
            window_duration_s=cfg.decode.window_duration_s,
        )
        transform = aligner.compute_timestamp_transform()
        record["alignment_valid"] = (len(transform) == cfg.alignment.temporal_bins)
    except Exception as exc:
        record["alignment_valid"] = False

    return record


def main() -> int:
    """Run data audit entry point."""
    args = parse_args()
    setup_logging(level=logging.INFO)
    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info("Stage 1-3 Data Audit: Inspection, Manifests, Split Exclusivity")
    logger.info("=" * 70)

    cfg = load_data_config(config_path=args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve only user-provided local dataset locations from CLI or config.
    dataset_roots: dict[str, Path] = {}
    if args.dataset_root:
        dataset_roots["fakeavceleb"] = Path(args.dataset_root).expanduser().resolve()
    else:
        for name, dataset in cfg.datasets.items():
            root_value = dataset.get("root") if isinstance(dataset, dict) else getattr(dataset, "root", None)
            if root_value and "${" not in str(root_value):
                root = Path(root_value).expanduser()
                if root.is_dir():
                    dataset_roots[name.lower()] = root.resolve()

    for name, root in dataset_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Configured {name} dataset root does not exist: {root}")

    # Build source and split manifests before any training if licensed local data exists.
    split_results: dict[str, Any] = {}
    source_table_path = output_dir / "source_table.csv"
    if dataset_roots:
        source_table = build_source_table(
            dataset_roots,
            output_path=source_table_path,
            compute_hashes=not args.skip_file_hashes,
        )
        if len(source_table) == 0:
            logger.warning("Source table is empty. Skipping split generation.")
            split_results = {}
        else:
            validate_source_table(source_table)
            split_results = create_identity_disjoint_split(
                source_table,
                train_ratio=cfg.split.train,
                val_ratio=cfg.split.val,
                test_ratio=cfg.split.test,
                seed=args.seed,
                output_dir=output_dir / "splits",
            )
            if split_results and any(len(df) > 0 for df in split_results.values()):
                verify_split_integrity(split_results)
                check_cross_partition_duplicates(split_results)
                logger.info("Created source table and split manifests under %s", output_dir)
            else:
                logger.warning("Split manifests are empty; skipping integrity verification.")
    else:
        logger.info("No configured licensed dataset root; no dataset manifests were generated.")

    # Collect sample clips for audit inspection
    clips_to_audit: list[Path] = []
    if dataset_roots:
        logger.info("Scanning local dataset roots for audit clips.")
        for dataset_root in dataset_roots.values():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
                clips_to_audit.extend(dataset_root.rglob(ext))
                if len(clips_to_audit) >= args.max_clips:
                    break
            if len(clips_to_audit) >= args.max_clips:
                break
        clips_to_audit = clips_to_audit[:args.max_clips]
        logger.info("Found %d clips for audit inspection.", len(clips_to_audit))
    else:
        logger.info("Using the permitted frame/audio fixtures for a decode smoke check only.")
        fixture_video = Path("tests/fixtures/test_clip.npz")
        fixture_audio = Path("tests/fixtures/test_tone.wav")
        if fixture_video.is_file():
            clips_to_audit.append(fixture_video)

    # Audit each clip
    audit_results: list[dict[str, Any]] = []
    fixture_audio = Path("tests/fixtures/test_tone.wav") if not dataset_roots else None
    for clip_path in clips_to_audit:
        res = audit_clip(clip_path, fixture_audio, cfg)
        audit_results.append(res)

    # Quantify audit statistics
    total_audited = len(audit_results)
    if total_audited > 0:
        valid_video = sum(1 for r in audit_results if r.get("video_valid", False))
        valid_audio = sum(1 for r in audit_results if r.get("audio_valid", False))
        clipped = sum(1 for r in audit_results if r.get("has_clipping", False))

        logger.info("-" * 70)
        logger.info("Audit Inspection Summary (%d clips sampled):", total_audited)
        logger.info("  Valid video windows: %d / %d (%.1f%%)", valid_video, total_audited, 100.0 * valid_video / total_audited)
        logger.info("  Valid audio tracks:  %d / %d (%.1f%%)", valid_audio, total_audited, 100.0 * valid_audio / total_audited)
        logger.info("  Clipped audio files: %d / %d (%.1f%%)", clipped, total_audited, 100.0 * clipped / total_audited)
        logger.info("-" * 70)
    else:
        logger.info("Audit executed cleanly with 0 active clip fixtures.")

    split_counts: dict[str, int] = {}
    if split_results:
        for name, df in split_results.items():
            split_counts[name] = len(df) if hasattr(df, "__len__") else 0

    report = {
        "seed": args.seed,
        "licensed_dataset_roots": {name: str(root) for name, root in dataset_roots.items()},
        "source_table": str(source_table_path) if dataset_roots else None,
        "split_counts": split_counts,
        "audited_clips": audit_results,
        "dataset_manifests_generated": bool(dataset_roots),
    }
    report_path = output_dir / "data_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if dataset_roots:
        logger.info("Data audit and manifest generation complete: %s", report_path)
    else:
        logger.info(
            "Fixture decode smoke check complete: %s. No licensed dataset manifests were generated.",
            report_path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
