"""Evaluates on the manipulation family held out before training, reporting per-family mean +/- bootstrap 95% CI.

Status: implemented — provides evaluate_cross_manipulation() for evaluation on held-out
manipulation families with bootstrap confidence intervals. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate.metrics import bootstrap_confidence_interval, compute_discrimination_metrics

logger = logging.getLogger(__name__)


@dataclass
class CrossManipulationResult:
    """Cross-manipulation evaluation results.

    Attributes
    ----------
    held_out_family:
        The held-out manipulation family.
    seen_metrics:
        Metrics on seen manipulation families (in-domain test).
    held_out_metrics:
        Metrics on the held-out family.
    per_family_results:
        Per-family breakdown.
    confidence_intervals:
        Bootstrap 95% CIs for key metrics.
    """
    held_out_family: str
    seen_metrics: dict[str, Any] = field(default_factory=dict)
    held_out_metrics: dict[str, Any] = field(default_factory=dict)
    per_family_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def evaluate_cross_manipulation(
    targets: np.ndarray,
    probabilities: np.ndarray,
    manipulation_families: np.ndarray,
    held_out_family: str,
    threshold: float | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> CrossManipulationResult:
    """Evaluate cross-manipulation generalization.

    Per Section 9.2: separate in-domain and held-out results.
    Never mix held-out results with in-domain results.

    Parameters
    ----------
    targets:
        Binary ground truth ``(N,)``.
    probabilities:
        Predicted probabilities ``(N,)``.
    manipulation_families:
        Per-sample manipulation family labels ``(N,)``.
    held_out_family:
        The family that was held out before training.
    threshold:
        Decision threshold (from validation).
    n_bootstrap:
        Number of bootstrap iterations for CIs.
    confidence_level:
        Confidence level for bootstrap CIs.
    seed:
        Random seed.

    Returns
    -------
    CrossManipulationResult
        Complete cross-manipulation evaluation.
    """
    import warnings

    targets = np.asarray(targets).flatten()
    probabilities = np.asarray(probabilities).flatten()
    families = np.asarray(manipulation_families).flatten()

    if len(targets) != len(probabilities) or len(targets) != len(families):
        raise ValueError(
            f"Mismatched input lengths: targets={len(targets)}, "
            f"probabilities={len(probabilities)}, families={len(families)}"
        )

    # Split by held-out vs seen
    held_out_mask = families == held_out_family
    seen_mask = ~held_out_mask

    if held_out_mask.sum() == 0:
        msg = f"Held-out family '{held_out_family}' not found in provided manipulation families."
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    result = CrossManipulationResult(held_out_family=held_out_family)

    # Seen families (in-domain test)
    if seen_mask.sum() > 0:
        seen_metrics = compute_discrimination_metrics(
            targets[seen_mask], probabilities[seen_mask], threshold=threshold
        )
        result.seen_metrics = seen_metrics.to_dict()

    # Held-out family
    if held_out_mask.sum() > 0:
        ho_metrics = compute_discrimination_metrics(
            targets[held_out_mask], probabilities[held_out_mask], threshold=threshold
        )
        result.held_out_metrics = ho_metrics.to_dict()

        # Bootstrap CI for held-out (guarded against single-class case)
        if len(np.unique(targets[held_out_mask])) > 1:
            for metric_name in ["roc_auc", "pr_auc", "balanced_accuracy"]:
                point, lower, upper = bootstrap_confidence_interval(
                    targets[held_out_mask],
                    probabilities[held_out_mask],
                    metric_fn=metric_name,
                    n_iterations=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                    threshold=threshold if threshold is not None else 0.5,
                )
                result.confidence_intervals[f"held_out_{metric_name}"] = (
                    point, lower, upper
                )
        else:
            logger.warning(
                "Held-out family '%s' contains only one class. Bootstrap CIs are undefined.",
                held_out_family,
            )

    # Per-family breakdown (guarded against single-class family subsets)
    for family in np.unique(families):
        fam_mask = families == family
        if fam_mask.sum() > 0:
            fam_targets = targets[fam_mask]
            fam_probs = probabilities[fam_mask]
            if len(np.unique(fam_targets)) < 2:
                logger.info(
                    "Family '%s' contains only one class (%s). Discriminating AUC metrics are undefined.",
                    family,
                    np.unique(fam_targets).tolist(),
                )
            fam_metrics = compute_discrimination_metrics(
                fam_targets, fam_probs, threshold=threshold
            )
            result.per_family_results[str(family)] = fam_metrics.to_dict()

    logger.info(
        "Cross-manipulation eval: held_out='%s' AUROC=%.4f, seen AUROC=%.4f",
        held_out_family,
        result.held_out_metrics.get("roc_auc", 0.0),
        result.seen_metrics.get("roc_auc", 0.0),
    )

    return result
