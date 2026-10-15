"""Brier score, Expected Calibration Error (15 equal-mass bins), reliability diagrams, and validation-only temperature scaling.

Status: implemented — provides compute_calibration_metrics(), TemperatureScaler
(fitted on validation only), and plot_reliability_diagram(). Raw and calibrated
metrics are kept distinct per Section 9.3. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration metrics per Section 9.3.

    Attributes
    ----------
    brier_score:
        Mean squared error between probabilities and targets.
    ece:
        Expected Calibration Error with equal-mass bins.
    nll:
        Negative log-likelihood.
    n_bins:
        Number of bins used for ECE.
    bin_accuracies:
        Per-bin accuracy for reliability diagram.
    bin_confidences:
        Per-bin mean predicted probability.
    bin_counts:
        Per-bin sample count.
    """
    brier_score: float = 0.0
    ece: float = 0.0
    nll: float = 0.0
    n_bins: int = 15
    bin_accuracies: list[float] | None = None
    bin_confidences: list[float] | None = None
    bin_counts: list[int] | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict for logging."""
        return {
            "brier_score": self.brier_score,
            "ece": self.ece,
            "nll": self.nll,
        }


def compute_calibration_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Compute calibration metrics per Section 9.3.

    Uses equal-mass (quantile) bins for ECE.

    Parameters
    ----------
    targets:
        Binary ground truth ``(N,)``.
    probabilities:
        Predicted probabilities ``(N,)``.
    n_bins:
        Number of equal-mass bins for ECE (default 15).

    Returns
    -------
    CalibrationMetrics
        Computed calibration metrics.
    """
    targets = np.asarray(targets).flatten().astype(float)
    probs = np.asarray(probabilities).flatten()
    n = len(targets)

    # Brier score
    brier = float(np.mean((probs - targets) ** 2))

    # Negative log-likelihood
    eps = 1e-7
    nll = float(-np.mean(
        targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps)
    ))

    # ECE with equal-mass bins (quantile-based)
    sorted_idx = np.argsort(probs)
    bin_size = max(1, n // n_bins)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        if start >= n:
            break

        bin_idx = sorted_idx[start:end]
        bin_targets = targets[bin_idx]
        bin_probs = probs[bin_idx]
        bin_n = len(bin_idx)

        acc = float(np.mean(bin_targets))
        conf = float(np.mean(bin_probs))
        bin_accuracies.append(acc)
        bin_confidences.append(conf)
        bin_counts.append(bin_n)

        ece += (bin_n / n) * abs(acc - conf)

    return CalibrationMetrics(
        brier_score=brier,
        ece=float(ece),
        nll=nll,
        n_bins=n_bins,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


class TemperatureScaler(nn.Module):
    """Temperature scaling for post-hoc calibration.

    Learns a single temperature parameter on the VALIDATION set only
    (Section 8, stage 7).  Test results must report both raw and scaled metrics.

    Parameters
    ----------
    initial_temperature:
        Starting temperature value (default 1.5).
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([initial_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature.

        Parameters
        ----------
        logits:
            Raw logits ``(N, 1)`` or ``(N,)``.

        Returns
        -------
        torch.Tensor
            Temperature-scaled logits.
        """
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature on validation logits.

        Parameters
        ----------
        val_logits:
            Validation logits ``(N, 1)`` or ``(N,)``.
        val_targets:
            Validation binary targets ``(N,)`` or ``(N, 1)``.
        lr:
            Learning rate for LBFGS optimizer.
        max_iter:
            Maximum iterations.

        Returns
        -------
        float
            Learned temperature value.
        """
        val_logits = val_logits.detach().float()
        val_targets = val_targets.detach().float().view_as(val_logits)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = nll_criterion(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        learned_temp = float(self.temperature.item())
        logger.info("Temperature scaling fitted: T=%.4f", learned_temp)
        return learned_temp


def plot_reliability_diagram(
    cal_metrics: CalibrationMetrics,
    save_path: str | None = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot a reliability diagram from calibration metrics.

    Parameters
    ----------
    cal_metrics:
        Computed calibration metrics with bin data.
    save_path:
        Path to save the figure (optional).
    title:
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping reliability diagram")
        return

    if cal_metrics.bin_accuracies is None:
        logger.warning("No bin data available for reliability diagram")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    accs = cal_metrics.bin_accuracies
    confs = cal_metrics.bin_confidences

    ax.bar(confs, accs, width=0.05, alpha=0.7, label="Model", edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{title}\nECE={cal_metrics.ece:.4f}, Brier={cal_metrics.brier_score:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Reliability diagram saved: %s", save_path)
    plt.close(fig)
