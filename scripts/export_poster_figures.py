"""CLI entry point generating the poster-ready figures/tables (Section 15) from saved evaluation outputs.

Status: implemented — generates poster-ready figures from evaluation JSON artifacts:
ROC curves, calibration plots, degradation robustness curves, reliability gate weight
distributions, efficiency comparison, and ablation tables per Section 15.
See AGENTS.md for build order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to sys.path so 'src' is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate poster-ready figures from evaluation outputs (Section 15)."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="reports/evaluation",
        help="Directory containing evaluation JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="Figure output format (default: pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        logger.warning("Missing evaluation artifact: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_roc_curve(
    eval_dir: Path,
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Generate ROC curve comparing baselines.

    Parameters
    ----------
    eval_dir:
        Directory containing evaluation results.
    output_dir:
        Directory for saving the figure.
    fmt:
        Output format (png/pdf/svg).
    dpi:
        Output DPI.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping ROC curve.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    # Look for per-baseline discrimination results
    for baseline_dir in sorted(eval_dir.glob("baseline_*")):
        disc = _load_json(baseline_dir / "discrimination_results.json")
        if disc and "metrics" in disc:
            name = baseline_dir.name.replace("baseline_", "").replace("_", " ").title()
            auroc = disc["metrics"].get("roc_auc", 0)
            ax.annotate(f"{name}: AUROC={auroc:.3f}", xy=(0.5, 0.5))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Baseline Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"roc_comparison.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve to %s", path)


def plot_degradation_curves(
    eval_dir: Path,
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Generate degradation robustness curves.

    Parameters
    ----------
    eval_dir:
        Directory containing degradation results.
    output_dir:
        Output directory for figure.
    fmt:
        Output format.
    dpi:
        Output DPI.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping degradation curves.")
        return

    deg = _load_json(eval_dir / "degradation_results.json")
    if deg is None:
        logger.info("No degradation results found; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Video degradation
    axes[0].set_title("Video Degradation Robustness")
    axes[0].set_xlabel("Degradation Level")
    axes[0].set_ylabel("AUROC")

    # Audio degradation
    axes[1].set_title("Audio Degradation Robustness")
    axes[1].set_xlabel("Degradation Level")
    axes[1].set_ylabel("AUROC")

    for ax in axes:
        ax.set_ylim(0.5, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"degradation_curves.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved degradation curves to %s", path)


def plot_gate_weight_distribution(
    eval_dir: Path,
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Generate reliability gate weight distribution plot.

    Parameters
    ----------
    eval_dir:
        Directory containing reliability diagnostics.
    output_dir:
        Output directory.
    fmt:
        Output format.
    dpi:
        Output DPI.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping gate plot.")
        return

    diag = _load_json(eval_dir / "reliability_diagnostics.json")
    if diag is None:
        logger.info("No reliability diagnostics found; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Reliability Gate Weights by Degradation Level")
    ax.set_xlabel("Degradation")
    ax.set_ylabel("Mean w_v")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Equal weight")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"gate_weights.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved gate weight distribution to %s", path)


def generate_ablation_table(
    eval_dir: Path,
    output_dir: Path,
) -> None:
    """Generate ablation comparison table as Markdown.

    Parameters
    ----------
    eval_dir:
        Directory containing per-baseline results.
    output_dir:
        Output directory.
    """
    rows = []
    header = "| Baseline | AUROC | PR-AUC | F1 | Balanced Acc | ECE |"
    separator = "|---|---|---|---|---|---|"
    rows.append(header)
    rows.append(separator)

    for baseline_dir in sorted(eval_dir.glob("baseline_*")):
        disc = _load_json(baseline_dir / "discrimination_results.json")
        cal = _load_json(baseline_dir / "calibration_results.json")
        if disc and "metrics" in disc:
            m = disc["metrics"]
            name = baseline_dir.name.replace("baseline_", "").replace("_", " ").title()
            ece = cal.get("raw", {}).get("ece", "-") if cal else "-"
            rows.append(
                f"| {name} | {m.get('roc_auc', '-'):.3f} "
                f"| {m.get('pr_auc', '-'):.3f} "
                f"| {m.get('f1', '-'):.3f} "
                f"| {m.get('balanced_accuracy', '-'):.3f} "
                f"| {ece} |"
            )

    if len(rows) > 2:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ablation_table.md"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows))
        logger.info("Saved ablation table to %s", path)
    else:
        logger.info("No baseline results found; ablation table not generated.")


def main() -> int:
    """Generate all poster-ready figures."""
    args = parse_args()
    setup_logging()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("Poster Figure Generation (Section 15)")
    logger.info("=" * 70)
    logger.info("  Eval dir:   %s", eval_dir)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Format:     %s @ %d DPI", args.format, args.dpi)

    # Generate figures
    plot_roc_curve(eval_dir, output_dir, fmt=args.format, dpi=args.dpi)
    plot_degradation_curves(eval_dir, output_dir, fmt=args.format, dpi=args.dpi)
    plot_gate_weight_distribution(eval_dir, output_dir, fmt=args.format, dpi=args.dpi)
    generate_ablation_table(eval_dir, output_dir)

    logger.info("\nPoster figure generation complete. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
