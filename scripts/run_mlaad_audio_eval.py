"""Run an exploratory audio-only transfer probe on the German MLAAD-tiny slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest_dataset import ManifestAVDataset, dataset_roots_from_config
from src.evaluate.cross_dataset_eval import evaluate_cross_dataset
from src.evaluate.mlaad_audio import (
    MlaadAudioDataset,
    bootstrap_grouped_auc,
    build_mlaad_audio_records,
    predict_audio_head,
)
from src.evaluate.metrics import compute_discrimination_metrics, find_optimal_threshold
from src.models.full_model import FullModel
from src.utils.config_loader import (
    load_data_config,
    load_eval_config,
    load_model_config,
    load_train_config,
    load_yaml,
)
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import set_seed

logger = logging.getLogger("run_mlaad_audio_eval")


def parse_args() -> argparse.Namespace:
    """Parse configuration, checkpoint, and optional smoke-run limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--manifest-dir", default="manifests/splits")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="reports/external_audio")
    parser.add_argument("--language", choices=("de", "en"), default=None)
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Optional source-group limit for a smoke run; limited outputs are not benchmark results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert logits into stable sigmoid probabilities."""
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _sha256(path: Path) -> str:
    """Hash a model checkpoint in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_fakeav_validation_audio(
    model: torch.nn.Module,
    dataset: ManifestAVDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Get the source-validation audio-head outputs used to select the fixed threshold."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
    )
    return predict_audio_head(model, loader, device)


def _load_model(checkpoint_path: Path, model_config: Any, device: torch.device) -> torch.nn.Module:
    """Load either a trained audio baseline or full model for transfer scoring."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("audio_encoder.") for key in state):
        # Full checkpoints contain visual weights too; don't fetch ImageNet
        # weights before replacing all parameters from this checkpoint.
        model_config.visual_encoder.pretrained = False
        model: torch.nn.Module = FullModel(model_config)
    elif any(key.startswith("encoder.") for key in state) and any(
        key.startswith("head.") for key in state
    ):
        from src.train.train_unimodal import UnimodalAudioModel

        model = UnimodalAudioModel(model_config)
    else:
        raise ValueError(
            f"Unsupported checkpoint format at {checkpoint_path}; expected an audio-only or full-model state dict"
        )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass-like outputs into JSON-native values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    """Fit the operating threshold on FakeAVCeleb validation and evaluate MLAAD once."""
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    if args.limit_groups is not None and args.limit_groups < 1:
        raise ValueError("--limit-groups must be positive")

    data_config = load_data_config(config_path=args.data_config)
    raw_data_config = load_yaml(args.data_config)
    external_spec = raw_data_config.get("external_datasets", {}).get("mlaad_tiny", {})
    if not external_spec:
        raise ValueError(f"No external_datasets.mlaad_tiny section in {args.data_config}")
    eval_config = load_eval_config(config_path=args.eval_config)
    eval_spec = eval_config.mlaad_tiny
    if not eval_spec.get("enabled", True):
        raise RuntimeError("MLAAD-tiny evaluation is disabled in the evaluation config")

    root = Path(args.dataset_root or external_spec["root"]).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    language = (args.language or eval_spec.get("language", external_spec.get("language", "de"))).lower()
    records = build_mlaad_audio_records(root, language)
    full_group_count = len({record.source_group for record in records})
    if args.limit_groups is not None:
        labels_by_group: dict[str, set[int]] = {}
        for record in records:
            labels_by_group.setdefault(record.source_group, set()).add(record.target)
        paired_groups = sorted(
            group for group, labels in labels_by_group.items() if labels == {0, 1}
        )
        if len(paired_groups) < args.limit_groups:
            raise ValueError(
                f"Requested {args.limit_groups} smoke groups, but only {len(paired_groups)} "
                "source groups have both local bona-fide and spoof audio"
            )
        selected = set(paired_groups[: args.limit_groups])
        records = [record for record in records if record.source_group in selected]

    roots = dataset_roots_from_config(data_config)
    if "fakeavceleb" not in roots:
        raise RuntimeError("Threshold selection requires the local FakeAVCeleb validation split")
    train_config = load_train_config(config_path=args.train_config)
    validation_data = ManifestAVDataset(
        Path(args.manifest_dir) / "val_manifest.csv",
        roots,
        Path(data_config.cache.cache_dir) / "features",
        data_config,
        allowed_datasets={"fakeavceleb"},
    )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, load_model_config(config_path=args.model_config), device)
    val = _predict_fakeav_validation_audio(
        model,
        validation_data,
        batch_size=int(train_config.batch_size),
        num_workers=int(train_config.num_workers),
        device=device,
    )
    val_probs = _probabilities(val["logits"])
    threshold_method = str(eval_config.threshold.get("method", "youden_j"))
    threshold = find_optimal_threshold(val["targets"], val_probs, method=threshold_method)

    external_dataset = MlaadAudioDataset(
        records,
        data_config,
        start_seconds=float(eval_spec.get("start_seconds", 0.0)),
    )
    external_loader = torch.utils.data.DataLoader(
        external_dataset,
        batch_size=int(train_config.batch_size),
        shuffle=False,
        num_workers=max(0, int(train_config.num_workers)),
        pin_memory=device.type == "cuda",
    )
    external = predict_audio_head(model, external_loader, device)
    external_probs = _probabilities(external["logits"])
    result = evaluate_cross_dataset(
        external["targets"],
        external_probs,
        source_dataset="FakeAVCeleb",
        target_dataset="MLAAD-tiny",
        threshold=threshold,
    )

    per_generator: dict[str, Any] = {}
    generators = set(external["generators"]) - {"bona_fide"}
    for generator in sorted(generators):
        mask = (external["targets"] == 0) | (
            (external["targets"] == 1) & (external["generators"] == generator)
        )
        per_generator[generator] = compute_discrimination_metrics(
            external["targets"][mask],
            external_probs[mask],
            threshold=threshold,
        ).to_dict()

    bootstrap = bootstrap_grouped_auc(
        external["targets"],
        external_probs,
        external["source_groups"],
        n_iterations=int(eval_spec.get("bootstrap_iterations", 1000)),
        confidence_level=float(eval_config.bootstrap.get("confidence_level", 0.95)),
        seed=args.seed,
    )
    real_groups = set(external["source_groups"][external["targets"] == 0])
    fake_groups = external["source_groups"][external["targets"] == 1]
    fake_rows_without_source = int(sum(group not in real_groups for group in fake_groups))
    fake_groups_with_source = len(set(fake_groups) & real_groups)
    report = {
        "evaluation_scope": "audio_only_exploratory_label_semantic_transfer",
        "training_dataset": "FakeAVCeleb",
        "target_dataset": "MLAAD-tiny",
        "source_label_semantics": "clip_fake",
        "target_label_semantics": "bona_fide_vs_spoof_audio",
        "interpretation": "exploratory_label_semantic_transfer_probe",
        "target_language": language,
        "dataset_revision": str(external_spec.get("revision", "unspecified")),
        "license_note": str(external_spec.get("license", "")),
        "threshold_source": "FakeAVCeleb validation audio head",
        "threshold_method": threshold_method,
        "validation_threshold": float(threshold),
        "test_threshold_retuned": False,
        "transfer_type": result.transfer_type,
        "compatibility_checks": result.compatibility_checks,
        "n_audio_windows": int(len(external["targets"])),
        "n_spoof_windows": int(np.sum(external["targets"] == 1)),
        "n_bona_fide_windows": int(np.sum(external["targets"] == 0)),
        "n_source_utterances": int(len(set(external["source_groups"]))),
        "n_spoof_source_groups_with_local_bona_fide_audio": int(fake_groups_with_source),
        "n_spoof_windows_without_local_source_audio": fake_rows_without_source,
        "n_source_utterances_full_language_slice": int(full_group_count),
        "smoke_limit_groups": args.limit_groups,
        "is_full_language_slice": args.limit_groups is None,
        "metrics": result.metrics,
        "roc_auc_group_bootstrap_95ci": bootstrap,
        "per_generator_metrics": per_generator,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mlaad_tiny_{language}_audio_eval.json"
    output_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    logger.info("Saved separate audio-only transfer report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
