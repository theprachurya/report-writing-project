"""Evaluate a selected checkpoint on validation and the locked in-domain test split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest_dataset import ManifestAVDataset, dataset_roots_from_config
from src.evaluate.calibration import TemperatureScaler, compute_calibration_metrics
from src.evaluate.metrics import bootstrap_confidence_interval, compute_discrimination_metrics, find_optimal_threshold
from src.evaluate.reliability_diagnostics import analyze_reliability_behavior
from src.models.full_model import FullModel
from src.utils.config_loader import load_data_config, load_eval_config, load_model_config, load_train_config
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse paths to configs, manifests, checkpoint, and output directory."""
    parser = argparse.ArgumentParser(description="Evaluate a trained AV detector on the locked test split.")
    parser.add_argument("--config", default="configs/eval_config.yaml", help="Evaluation YAML configuration.")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--manifest-dir", default="manifests/splits")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/reliability_fusion/best_model.pt")
    parser.add_argument("--output-dir", default="reports/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(
    checkpoint_path: str | Path,
    model_config: object | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load an evaluated checkpoint; refuse to produce results from random weights."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = model_config or load_model_config()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train a reliability fusion model first.")
    model = FullModel(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)
    return model


def _predict(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Run one deterministic inference pass and retain the required diagnostics."""
    result: dict[str, list[torch.Tensor]] = {key: [] for key in ("logits", "targets", "w_v", "w_a")}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: (value.to(device).float() if isinstance(value, torch.Tensor) and value.is_floating_point() else (value.to(device) if isinstance(value, torch.Tensor) else value))
                for key, value in batch.items()
            }
            output = model(
                frames=batch["frames"],
                mel_spec=batch["mel_spec"],
                quality_v=batch["quality_v"],
                quality_a=batch["quality_a"],
                visual_missing=batch["visual_missing"],
                audio_missing=batch["audio_missing"],
            )
            result["logits"].append(output["logit_f"].float().cpu().reshape(-1))
            result["targets"].append(batch["target"].float().cpu().reshape(-1))
            result["w_v"].append(output["w_v"].float().cpu().reshape(-1))
            result["w_a"].append(output["w_a"].float().cpu().reshape(-1))
    if not result["targets"]:
        raise ValueError("Evaluation split has no rows")
    return {key: torch.cat(values).numpy() for key, values in result.items()}


def _probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to finite probabilities with an optional fitted temperature."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -80.0, 80.0)))


def _jsonable(value: object) -> object:
    """Convert NumPy/dataclass report values into JSON-native objects."""
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
    """Fit validation-only calibration/thresholds, then evaluate test exactly once."""
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, load_model_config(config_path=args.model_config))
    device = next(model.parameters()).device
    data_config = load_data_config(config_path=args.data_config)
    train_config = load_train_config(config_path=args.train_config)
    eval_config = load_eval_config(config_path=args.config)
    roots = dataset_roots_from_config(data_config)
    if "fakeavceleb" not in roots:
        raise RuntimeError("In-domain audio-visual evaluation requires the licensed FakeAVCeleb root.")

    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(data_config.cache.cache_dir) / "features"
    val_data = ManifestAVDataset(
        manifest_dir / "val_manifest.csv", roots, cache_dir, data_config,
        allowed_datasets={"fakeavceleb"},
    )
    batch_size = int(train_config.batch_size)
    workers = max(0, int(train_config.num_workers))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Validation is the only source for model operating points and temperature.
    val = _predict(model, val_loader, device)
    val_probs = _probabilities(val["logits"])
    threshold_method = str(eval_config.threshold.get("method", "youden_j"))
    threshold_raw = find_optimal_threshold(val["targets"], val_probs, method=threshold_method)
    scaler = TemperatureScaler()
    scaler.fit(torch.tensor(val["logits"]), torch.tensor(val["targets"]))
    temperature = max(float(scaler.temperature.detach().cpu().item()), 1e-6)
    val_scaled = _probabilities(val["logits"], temperature)
    threshold_scaled = find_optimal_threshold(val["targets"], val_scaled, method=threshold_method)

    # The held-out test loader is traversed once after every validation decision is fixed.
    test_data = ManifestAVDataset(
        manifest_dir / "test_manifest.csv", roots, cache_dir, data_config,
        allowed_datasets={"fakeavceleb"},
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    test = _predict(model, test_loader, device)
    test_raw = _probabilities(test["logits"])
    test_scaled = _probabilities(test["logits"], temperature)
    results: dict[str, object] = {
        "validation": {
            "threshold_raw": float(threshold_raw),
            "threshold_scaled": float(threshold_scaled),
            "temperature": temperature,
            "raw_metrics": compute_discrimination_metrics(val["targets"], val_probs, threshold=threshold_raw).__dict__,
            "scaled_metrics": compute_discrimination_metrics(val["targets"], val_scaled, threshold=threshold_scaled).__dict__,
        },
        "test": {
            "n_samples": int(len(test["targets"])),
            "raw_metrics": compute_discrimination_metrics(test["targets"], test_raw, threshold=threshold_raw).__dict__,
            "scaled_metrics": compute_discrimination_metrics(test["targets"], test_scaled, threshold=threshold_scaled).__dict__,
            "raw_calibration": compute_calibration_metrics(
                test["targets"], test_raw,
                n_bins=int(eval_config.metrics.get("ece_bins", 15)),
            ).__dict__,
            "scaled_calibration": compute_calibration_metrics(
                test["targets"], test_scaled,
                n_bins=int(eval_config.metrics.get("ece_bins", 15)),
            ).__dict__,
            "reliability": analyze_reliability_behavior(
                test["w_v"], test["w_a"],
                collapse_threshold=eval_config.reliability_diagnostics.get("collapse_threshold", 0.95),
            ).__dict__,
        },
        "protocol": {
            "checkpoint": str(checkpoint_path),
            "seed": int(args.seed),
            "thresholds_and_temperature_fit_on": "validation",
            "test_inference_passes": 1,
            "raw_and_scaled_metrics_kept_separate": True,
        },
    }
    bootstrap_cfg = eval_config.bootstrap
    if bootstrap_cfg.get("enabled", False):
        # Group labels are drawn from the test manifest but do not affect threshold selection.
        result_ci = bootstrap_confidence_interval(
        test["targets"], test_raw,
            n_iterations=int(bootstrap_cfg.get("n_iterations", 1000)),
            confidence_level=float(bootstrap_cfg.get("confidence_level", 0.95)),
            seed=args.seed,
            stratify_by=test_data.rows["source_identity"].to_numpy(),
        )
        results["test"]["raw_auroc_bootstrap_ci"] = result_ci  # type: ignore[index]

    report_path = output_dir / "evaluation_results.json"
    report_path.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    logger.info("Clean held-out evaluation saved to %s", report_path)
    logger.info("This run covers clean in-domain evaluation; degradation/cross-dataset suites are separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
