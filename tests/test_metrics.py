"""Verifies metric implementations (AUROC, ECE, etc.) against known reference values.

Status: implemented — tests discrimination metrics (ROC-AUC, PR-AUC, F1, balanced
accuracy, confusion matrix), optimal threshold selection via Youden's J, Brier score,
ECE with equal-mass bins, and temperature scaling against known analytic reference values.
See AGENTS.md for build order.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluate.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    compute_calibration_metrics,
    compute_ece,
    compute_nll,
)
from src.evaluate.cross_dataset_eval import (
    CrossDatasetResult,
    check_label_compatibility,
    evaluate_cross_dataset,
)
from src.evaluate.cross_manipulation_eval import (
    CrossManipulationResult,
    evaluate_cross_manipulation,
)
from src.evaluate.degradation_suite import (
    DegradationSuite,
    DegradationSuiteResults,
    DegradationSuiteRunner,
)
from src.evaluate.efficiency_profiling import (
    EfficiencyProfile,
    measure_latency,
    measure_preprocessing_latency,
    profile_model,
)
from src.evaluate.metrics import (
    DiscriminationMetrics,
    bootstrap_confidence_interval,
    compute_discrimination_metrics,
    compute_prauc,
    find_optimal_threshold,
)
from src.evaluate.reliability_diagnostics import (
    ReliabilityDiagnostics,
    analyze_reliability_behavior,
    compute_gate_entropy,
)


def test_perfect_discrimination_metrics() -> None:
    """Perfect classifier achieves 1.0 across all discrimination metrics."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not yet installed")

    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Separable probabilities
    probs = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])

    metrics = compute_discrimination_metrics(targets, probs, threshold=0.5)

    assert isinstance(metrics, DiscriminationMetrics)
    assert np.isclose(metrics.roc_auc, 1.0)
    assert np.isclose(metrics.pr_auc, 1.0)
    assert np.isclose(metrics.balanced_accuracy, 1.0)
    assert np.isclose(metrics.f1, 1.0)
    assert np.isclose(metrics.precision, 1.0)
    assert np.isclose(metrics.recall, 1.0)
    assert np.isclose(metrics.specificity, 1.0)

    # Confusion matrix: [[TN, FP], [FN, TP]] -> [[4, 0], [0, 4]]
    cm = metrics.confusion_matrix
    cm_list = cm.tolist() if hasattr(cm, "tolist") else cm
    assert cm_list == [[4, 0], [0, 4]]

    # Ensure confusion_matrix is included in to_dict()
    metric_dict = metrics.to_dict()
    assert "confusion_matrix" in metric_dict
    assert metric_dict["confusion_matrix"] == [[4, 0], [0, 4]]


def test_inverted_discrimination_metrics() -> None:
    """Inverted classifier achieves 0.0 ROC-AUC."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not yet installed")

    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.1])

    metrics = compute_discrimination_metrics(targets, probs, threshold=0.5)
    assert np.isclose(metrics.roc_auc, 0.0)


def test_single_class_metrics_guard() -> None:
    """Single-class case guards against exceptions and returns 0.0."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not yet installed")

    targets = np.array([1, 1, 1, 1])
    probs = np.array([0.9, 0.8, 0.7, 0.6])

    pr_auc = compute_prauc(targets, probs)
    assert pr_auc == 0.0

    metrics = compute_discrimination_metrics(targets, probs)
    assert metrics.roc_auc == 0.0
    assert metrics.pr_auc == 0.0

    point, lower, upper = bootstrap_confidence_interval(
        targets, probs, metric_fn="roc_auc", n_iterations=10
    )
    assert point == 0.0


def test_bootstrap_confidence_interval_threshold() -> None:
    """bootstrap_confidence_interval accepts threshold for balanced accuracy."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not yet installed")

    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.2, 0.4, 0.6, 0.8])

    point_30, _, _ = bootstrap_confidence_interval(
        targets, probs, metric_fn="balanced_accuracy", threshold=0.3, n_iterations=20
    )
    point_50, _, _ = bootstrap_confidence_interval(
        targets, probs, metric_fn="balanced_accuracy", threshold=0.5, n_iterations=20
    )
    assert np.isclose(point_30, 0.75)
    assert np.isclose(point_50, 1.0)
    assert point_30 != point_50


def test_optimal_threshold_youden_j() -> None:
    """Youden's J selects a threshold that maximizes TPR - FPR clipped to [0, 1]."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not yet installed")

    # Real samples at 0.1, 0.3; Fake samples at 0.6, 0.8
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.3, 0.6, 0.8])

    threshold = find_optimal_threshold(targets, probs, method="youden_j")
    # Optimal threshold must separate [0.1, 0.3] from [0.6, 0.8] and be within [0.0, 1.0]
    assert 0.3 < threshold <= 0.6, f"Expected threshold in (0.3, 0.6], got {threshold}"
    assert 0.0 <= threshold <= 1.0


def test_known_brier_score() -> None:
    """Brier score matches exact analytical mean squared error."""
    targets = np.array([1.0, 0.0])
    probs = np.array([0.8, 0.2])

    # Expected: ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    calib = compute_calibration_metrics(targets, probs, n_bins=2)
    assert np.isclose(calib.brier_score, 0.04, atol=1e-6)


def test_compute_nll_clipped() -> None:
    """compute_nll clips probabilities avoiding log(0) NaN/Inf."""
    targets = np.array([1.0, 0.0])
    probs = np.array([1.0, 0.0])

    nll = compute_nll(targets, probs)
    assert np.isfinite(nll)
    assert nll >= 0.0


def test_known_ece_equal_mass() -> None:
    """ECE with equal-mass bins matches step-by-step quantile calculation."""
    # 4 samples, 2 equal-mass bins of size 2
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    # Bin 1: probs [0.1, 0.2] (mean=0.15), targets [0, 0] (mean=0.0) -> abs diff = 0.15
    # Bin 2: probs [0.8, 0.9] (mean=0.85), targets [1, 1] (mean=1.0) -> abs diff = 0.15
    # Overall ECE = 0.5 * 0.15 + 0.5 * 0.15 = 0.15
    calib = compute_calibration_metrics(targets, probs, n_bins=2)
    assert np.isclose(calib.ece, 0.15, atol=1e-5)

    ece_direct = compute_ece(targets, probs, n_bins=2, strategy="equal_mass")
    assert np.isclose(ece_direct, 0.15, atol=1e-5)


def test_temperature_scaler() -> None:
    """TemperatureScaler fits temperature on validation logits and scales properly."""
    scaler = TemperatureScaler(initial_temperature=1.0)

    val_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    val_targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    initial_loss = scaler.fit(val_logits, val_targets, max_iter=20, lr=0.05)
    learned_t = scaler.temperature.item()

    assert learned_t >= 1e-4, "Temperature must remain positive and >= 1e-4"
    scaled = scaler(val_logits)
    assert scaled.shape == val_logits.shape

    # Test clamping on negative temperature parameter
    with torch.no_grad():
        scaler.temperature.data.fill_(-2.0)
    clamped_scaled = scaler(val_logits)
    assert not torch.isnan(clamped_scaled).any()


def test_degradation_suite_conditions_and_run_all() -> None:
    """DegradationSuite includes Section 9.2 conditions and run_all executes safely."""
    suite = DegradationSuite()
    conditions = suite.get_degradation_conditions()
    cond_names = [c["name"] for c in conditions]

    assert any("frame_drop" in name for name in cond_names)
    assert any("packet_dropout" in name for name in cond_names)
    assert any("codec_resample" in name for name in cond_names)

    def dummy_eval(cond: dict):
        return 0.85, 0.6, 0.4

    results = suite.run_all(dummy_eval, clean_metric=0.90, clean_w_v=0.5)
    assert isinstance(results, DegradationSuiteResults)
    assert len(results.results) == len(conditions)

    # Safe division check with clean_metric == 0
    zero_res = suite.run_condition(conditions[0], dummy_eval, clean_metric=0.0)
    assert zero_res.relative_change == 0.0


def test_cross_dataset_validation_and_dfdc_warning() -> None:
    """evaluate_cross_dataset validates lengths, requires threshold, and warns on DFDC."""
    t = np.array([0, 1])
    p = np.array([0.2, 0.8])

    # Mismatched lengths
    with pytest.raises(ValueError, match="match"):
        evaluate_cross_dataset(t, np.array([0.2]), "fakeavceleb", "dfdc", threshold=0.5)

    # Missing threshold
    with pytest.raises(ValueError, match="pre-selected threshold"):
        evaluate_cross_dataset(t, p, "fakeavceleb", "dfdc", threshold=None)

    # Warning on DFDC
    with pytest.warns(UserWarning, match="visual-only"):
        res = evaluate_cross_dataset(t, p, "fakeavceleb", "dfdc", threshold=0.5)
        assert res.transfer_type == "visual_only"


def test_cross_manipulation_single_class_and_missing_family() -> None:
    """evaluate_cross_manipulation handles single-class family subsets and warns on missing held-out."""
    targets = np.array([0, 1, 1, 1])
    probs = np.array([0.1, 0.7, 0.8, 0.9])
    families = np.array(["deepfakes", "deepfakes", "faceswap", "faceswap"])

    # Missing held-out family triggers warning
    with pytest.warns(UserWarning, match="not found"):
        res = evaluate_cross_manipulation(
            targets, probs, families, held_out_family="nonexistent_family", threshold=0.5
        )
        assert res.held_out_metrics == {}

    # faceswap has only class 1 (single-class guard)
    res2 = evaluate_cross_manipulation(
        targets, probs, families, held_out_family="faceswap", threshold=0.5, n_bootstrap=10
    )
    assert "faceswap" in res2.per_family_results


def test_efficiency_profiling_percentiles_and_preprocessing() -> None:
    """Efficiency profiling reports p50/p95 and separates preprocessing."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    sample_input = {"x": torch.randn(1, 10)}

    p50, p95, tp = measure_latency(model, sample_input, n_warmup=2, n_measure=10)
    assert p50 > 0
    assert p95 >= p50

    def dummy_preprocess(raw):
        return {"x": torch.tensor(raw, dtype=torch.float32)}

    prep_p50, prep_p95 = measure_preprocessing_latency(
        dummy_preprocess, np.zeros((1, 10)), n_warmup=2, n_measure=10
    )
    assert prep_p50 >= 0
    assert prep_p95 >= prep_p50

    profile = profile_model(
        model,
        sample_input=sample_input,
        n_measure=10,
        preprocess_fn=dummy_preprocess,
        sample_raw_input=np.zeros((1, 10)),
    )
    assert profile.includes_preprocessing
    assert profile.preprocessing_latency_p50_ms >= 0


def test_reliability_diagnostics_std_w_a_entropy_safe_spearmanr() -> None:
    """Reliability diagnostics tracks std_w_a, computes exact entropy, and handles constant inputs."""
    w_v = np.array([0.6, 0.7, 0.8])
    w_a = np.array([0.4, 0.3, 0.2])

    diag = analyze_reliability_behavior(
        w_v,
        w_a,
        severity_levels=np.array([1, 1, 1]),  # Constant severity: safe Spearman
        unimodal_losses_v=np.array([0.5, 0.5, 0.5]),
        unimodal_losses_a=np.array([0.5, 0.5, 0.5]),
    )

    assert hasattr(diag, "std_w_a")
    assert "std_w_a" in diag.to_dict()
    assert not np.isnan(diag.degradation_correlations["w_v_vs_severity"])

    # Entropy checks
    h_pure = compute_gate_entropy(np.array([1.0]), np.array([0.0]))
    assert np.isclose(h_pure, 0.0, atol=1e-5)

    h_uniform = compute_gate_entropy(np.array([0.5]), np.array([0.5]))
    assert np.isclose(h_uniform, np.log(2.0), atol=1e-5)
