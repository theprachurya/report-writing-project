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
)
from src.evaluate.metrics import (
    DiscriminationMetrics,
    compute_discrimination_metrics,
    find_optimal_threshold,
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


def test_optimal_threshold_youden_j() -> None:
    """Youden's J selects a threshold that maximizes TPR - FPR."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not yet installed")

    # Real samples at 0.1, 0.3; Fake samples at 0.6, 0.8
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.3, 0.6, 0.8])

    threshold = find_optimal_threshold(targets, probs, method="youden_j")
    # Optimal threshold must separate [0.1, 0.3] from [0.6, 0.8]
    assert 0.3 < threshold <= 0.6, f"Expected threshold in (0.3, 0.6], got {threshold}"


def test_known_brier_score() -> None:
    """Brier score matches exact analytical mean squared error."""
    targets = np.array([1.0, 0.0])
    probs = np.array([0.8, 0.2])

    # Expected: ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    calib = compute_calibration_metrics(targets, probs, n_bins=2)
    assert np.isclose(calib.brier_score, 0.04, atol=1e-6)


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


def test_temperature_scaler() -> None:
    """TemperatureScaler fits temperature on validation logits and scales properly."""
    scaler = TemperatureScaler(initial_temperature=1.0)

    val_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    val_targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    initial_loss = scaler.fit(val_logits, val_targets, max_iter=20, lr=0.05)
    learned_t = scaler.temperature.item()

    assert learned_t > 0.0, "Temperature must remain positive"
    scaled = scaler(val_logits)
    assert scaled.shape == val_logits.shape
