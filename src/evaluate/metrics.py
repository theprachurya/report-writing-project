"""ROC-AUC, PR-AUC, balanced accuracy, F1, precision/recall, confusion matrix at a validation-selected threshold.

Status: implemented — provides compute_discrimination_metrics() and helpers for all
discrimination metrics specified in Section 9.3. Threshold is always selected on
validation, never test. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscriminationMetrics:
    """Discrimination metrics per Section 9.3.

    Attributes
    ----------
    roc_auc:
        Area under the ROC curve (primary metric).
    pr_auc:
        Area under the Precision-Recall curve.
    balanced_accuracy:
        Average of per-class recall.
    f1:
        F1 score at the chosen threshold.
    precision:
        Precision at the chosen threshold.
    recall:
        Recall (sensitivity) at the chosen threshold.
    specificity:
        Specificity (true negative rate) at the chosen threshold.
    confusion_matrix:
        2×2 confusion matrix [[TN, FP], [FN, TP]].
    threshold:
        The decision threshold used (from validation).
    """
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    balanced_accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    threshold: float = 0.5

    def to_dict(self) -> dict[str, float]:
        """Convert to a flat dictionary for logging."""
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "threshold": self.threshold,
        }


def find_optimal_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    method: str = "youden_j",
) -> float:
    """Find the optimal decision threshold on a VALIDATION set.

    Parameters
    ----------
    targets:
        Binary ground truth ``(N,)``.
    probabilities:
        Predicted probabilities ``(N,)``.
    method:
        ``"youden_j"`` (default) uses Youden's J statistic.

    Returns
    -------
    float
        Optimal threshold.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(targets, probabilities)

    if method == "youden_j":
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return float(thresholds[best_idx])
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def compute_discrimination_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
    threshold_method: str = "youden_j",
) -> DiscriminationMetrics:
    """Compute all discrimination metrics per Section 9.3.

    Parameters
    ----------
    targets:
        Binary ground truth ``(N,)`` — 0 = real, 1 = fake.
    probabilities:
        Predicted probabilities ``(N,)``.
    threshold:
        Decision threshold.  If ``None``, computed from the data using
        *threshold_method*.  **Must come from validation, never test.**
    threshold_method:
        Method for automatic threshold selection.

    Returns
    -------
    DiscriminationMetrics
        All computed metrics.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    targets = np.asarray(targets).flatten()
    probabilities = np.asarray(probabilities).flatten()

    # ROC-AUC
    if len(set(targets)) > 1:
        roc_auc = float(roc_auc_score(targets, probabilities))
    else:
        roc_auc = 0.0
        logger.warning("Only one class present — ROC-AUC undefined.")

    # PR-AUC
    pr_auc = float(average_precision_score(targets, probabilities))

    # Find threshold
    if threshold is None:
        threshold = find_optimal_threshold(targets, probabilities, threshold_method)

    # Binary predictions
    predictions = (probabilities >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metrics at threshold
    bal_acc = float(balanced_accuracy_score(targets, predictions))
    f1 = float(f1_score(targets, predictions, zero_division=0))
    prec = float(precision_score(targets, predictions, zero_division=0))
    rec = float(recall_score(targets, predictions, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return DiscriminationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        balanced_accuracy=bal_acc,
        f1=f1,
        precision=prec,
        recall=rec,
        specificity=spec,
        confusion_matrix=cm.tolist(),
        threshold=threshold,
    )


def bootstrap_confidence_interval(
    targets: np.ndarray,
    probabilities: np.ndarray,
    metric_fn: str = "roc_auc",
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    stratify_by: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence intervals for a metric.

    Parameters
    ----------
    targets:
        Binary ground truth.
    probabilities:
        Predicted probabilities.
    metric_fn:
        Metric name (``"roc_auc"``, ``"pr_auc"``, ``"balanced_accuracy"``).
    n_iterations:
        Number of bootstrap samples.
    confidence_level:
        Confidence level (default 0.95).
    seed:
        Random seed.
    stratify_by:
        Optional group labels for stratified bootstrap.

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, lower_bound, upper_bound)``.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )

    rng = np.random.default_rng(seed)
    targets = np.asarray(targets).flatten()
    probabilities = np.asarray(probabilities).flatten()
    n = len(targets)
    groups = None if stratify_by is None else np.asarray(stratify_by).flatten()
    if groups is not None and len(groups) != n:
        raise ValueError("stratify_by must have one group label per sample")

    def _compute(t: np.ndarray, p: np.ndarray) -> float:
        if len(set(t)) < 2:
            return 0.0
        if metric_fn == "roc_auc":
            return float(roc_auc_score(t, p))
        elif metric_fn == "pr_auc":
            return float(average_precision_score(t, p))
        elif metric_fn == "balanced_accuracy":
            preds = (p >= 0.5).astype(int)
            return float(balanced_accuracy_score(t, preds))
        else:
            raise ValueError(f"Unknown metric: {metric_fn}")

    point_estimate = _compute(targets, probabilities)
    bootstrap_values = []

    for _ in range(n_iterations):
        if groups is None:
            idx = rng.choice(n, size=n, replace=True)
        else:
            unique_groups = np.unique(groups)
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        val = _compute(targets[idx], probabilities[idx])
        bootstrap_values.append(val)

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

    return point_estimate, lower, upper
