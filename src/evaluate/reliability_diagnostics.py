"""Analyzes gate weight (w_v, w_a) behavior vs degradation severity and unimodal loss; reports gate entropy and collapse detection.

Status: implemented — provides analyze_reliability_behavior() for diagnosing gate
weight shifts, correlations with degradation severity, and detecting gate collapse.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityDiagnostics:
    """Gate weight diagnostics per Section 9.3.

    Attributes
    ----------
    mean_w_v:
        Overall mean visual weight.
    mean_w_a:
        Overall mean audio weight.
    std_w_v:
        Standard deviation of visual weight.
    gate_entropy:
        Mean entropy of gate weight distribution.
    collapse_detected:
        Whether gate collapse is detected.
    collapse_details:
        Details about collapse (which modality dominates).
    degradation_correlations:
        Correlation between degradation severity and gate shift.
    per_condition_weights:
        Mean weights broken down by condition.
    """
    mean_w_v: float = 0.5
    mean_w_a: float = 0.5
    std_w_v: float = 0.0
    gate_entropy: float = 0.0
    collapse_detected: bool = False
    collapse_details: str = ""
    degradation_correlations: dict[str, float] = field(default_factory=dict)
    per_condition_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for logging."""
        return {
            "mean_w_v": self.mean_w_v,
            "mean_w_a": self.mean_w_a,
            "std_w_v": self.std_w_v,
            "gate_entropy": self.gate_entropy,
            "collapse_detected": self.collapse_detected,
            "collapse_details": self.collapse_details,
        }


def compute_gate_entropy(
    w_v: np.ndarray,
    w_a: np.ndarray,
) -> float:
    """Compute mean entropy of gate weight distribution.

    Higher entropy → more uniform (uncertain) weighting.
    Lower entropy → more decisive modality selection.

    Parameters
    ----------
    w_v:
        Visual weights ``(N,)``.
    w_a:
        Audio weights ``(N,)``.

    Returns
    -------
    float
        Mean entropy in nats.
    """
    eps = 1e-8
    entropy = -(
        w_v * np.log(w_v + eps) + w_a * np.log(w_a + eps)
    )
    return float(np.mean(entropy))


def detect_collapse(
    w_v: np.ndarray,
    w_a: np.ndarray,
    threshold: float = 0.95,
) -> tuple[bool, str]:
    """Detect if the gate has collapsed to always favoring one modality.

    Per Section 12: if gate always favors video/audio, report that outcome
    and adjust the claim, not the result.

    Parameters
    ----------
    w_v:
        Visual weights ``(N,)``.
    w_a:
        Audio weights ``(N,)``.
    threshold:
        If max(mean_w_v, mean_w_a) > threshold, flag collapse.

    Returns
    -------
    tuple[bool, str]
        ``(is_collapsed, details_message)``.
    """
    mean_v = float(np.mean(w_v))
    mean_a = float(np.mean(w_a))

    # Check if overwhelmingly one-sided
    fraction_v_dominant = float(np.mean(w_v > threshold))
    fraction_a_dominant = float(np.mean(w_a > threshold))

    if mean_v > threshold:
        return True, (
            f"Gate collapsed to visual: mean_w_v={mean_v:.4f}, "
            f"{fraction_v_dominant*100:.1f}% samples have w_v>{threshold}"
        )
    elif mean_a > threshold:
        return True, (
            f"Gate collapsed to audio: mean_w_a={mean_a:.4f}, "
            f"{fraction_a_dominant*100:.1f}% samples have w_a>{threshold}"
        )

    return False, "No collapse detected."


def analyze_reliability_behavior(
    w_v: np.ndarray,
    w_a: np.ndarray,
    conditions: np.ndarray | None = None,
    severity_levels: np.ndarray | None = None,
    unimodal_losses_v: np.ndarray | None = None,
    unimodal_losses_a: np.ndarray | None = None,
    collapse_threshold: float = 0.95,
) -> ReliabilityDiagnostics:
    """Full reliability behavior analysis.

    Parameters
    ----------
    w_v:
        Visual weights ``(N,)``.
    w_a:
        Audio weights ``(N,)``.
    conditions:
        Per-sample condition labels (e.g., degradation type).
    severity_levels:
        Per-sample severity levels for correlation.
    unimodal_losses_v:
        Per-sample visual unimodal losses.
    unimodal_losses_a:
        Per-sample audio unimodal losses.
    collapse_threshold:
        Threshold for collapse detection.

    Returns
    -------
    ReliabilityDiagnostics
        Complete diagnostics report.
    """
    w_v = np.asarray(w_v).flatten()
    w_a = np.asarray(w_a).flatten()

    diag = ReliabilityDiagnostics(
        mean_w_v=float(np.mean(w_v)),
        mean_w_a=float(np.mean(w_a)),
        std_w_v=float(np.std(w_v)),
        gate_entropy=compute_gate_entropy(w_v, w_a),
    )

    # Collapse detection
    collapsed, details = detect_collapse(w_v, w_a, collapse_threshold)
    diag.collapse_detected = collapsed
    diag.collapse_details = details
    if collapsed:
        logger.warning("Gate collapse detected: %s", details)

    # Per-condition weights
    if conditions is not None:
        conditions = np.asarray(conditions)
        for cond in np.unique(conditions):
            mask = conditions == cond
            diag.per_condition_weights[str(cond)] = {
                "mean_w_v": float(np.mean(w_v[mask])),
                "mean_w_a": float(np.mean(w_a[mask])),
                "n_samples": int(mask.sum()),
            }

    # Correlation with severity
    if severity_levels is not None:
        from scipy.stats import spearmanr
        severity = np.asarray(severity_levels).flatten()

        rho_v, p_v = spearmanr(severity, w_v)
        rho_a, p_a = spearmanr(severity, w_a)
        diag.degradation_correlations["w_v_vs_severity"] = float(rho_v)
        diag.degradation_correlations["w_a_vs_severity"] = float(rho_a)
        diag.degradation_correlations["w_v_pvalue"] = float(p_v)
        diag.degradation_correlations["w_a_pvalue"] = float(p_a)

    # Correlation with unimodal losses
    if unimodal_losses_v is not None and unimodal_losses_a is not None:
        from scipy.stats import spearmanr
        loss_v = np.asarray(unimodal_losses_v).flatten()
        loss_a = np.asarray(unimodal_losses_a).flatten()

        rho, _ = spearmanr(loss_v - loss_a, w_v - w_a)
        diag.degradation_correlations["loss_diff_vs_weight_diff"] = float(rho)

    logger.info(
        "Reliability diagnostics: mean_w_v=%.3f, mean_w_a=%.3f, entropy=%.3f, "
        "collapsed=%s",
        diag.mean_w_v, diag.mean_w_a, diag.gate_entropy, diag.collapse_detected,
    )

    return diag
