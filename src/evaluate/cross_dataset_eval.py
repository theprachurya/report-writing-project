"""Cross-dataset transfer evaluation with explicit checks that modality labels/licenses are compatible before any audio claim is made.

Status: implemented — provides evaluate_cross_dataset() with mandatory compatibility
checks. DFDC is treated as visual-transfer only unless audio semantics are audited
(Section 9.2). See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.evaluate.metrics import compute_discrimination_metrics

logger = logging.getLogger(__name__)


@dataclass
class CrossDatasetResult:
    """Cross-dataset evaluation results.

    Attributes
    ----------
    source_dataset:
        Dataset the model was trained on.
    target_dataset:
        Dataset being evaluated on.
    transfer_type:
        ``"full_av"`` or ``"visual_only"``.
    metrics:
        Discrimination metrics (AUROC with no threshold retuning).
    compatibility_checks:
        Record of compatibility verification.
    """
    source_dataset: str
    target_dataset: str
    transfer_type: str
    metrics: dict[str, Any] = field(default_factory=dict)
    compatibility_checks: dict[str, bool] = field(default_factory=dict)


# Datasets with verified audio-deepfake labels
_AUDIO_VERIFIED_DATASETS = {"fakeavceleb"}

# Datasets where audio labels are NOT verified for deepfake claims
_AUDIO_UNVERIFIED_DATASETS = {"dfdc"}


def check_label_compatibility(
    source_dataset: str,
    target_dataset: str,
    audio_audited: bool = False,
) -> dict[str, bool]:
    """Check whether cross-dataset evaluation is valid.

    Parameters
    ----------
    source_dataset:
        Training dataset name.
    target_dataset:
        Target evaluation dataset name.
    audio_audited:
        Whether target dataset audio labels have been explicitly audited.

    Returns
    -------
    dict[str, bool]
        Compatibility check results.
    """
    import warnings

    checks: dict[str, bool] = {}

    target_lower = target_dataset.lower()
    checks["visual_transfer_valid"] = True  # Visual transfer always valid

    if "dfdc" in target_lower or target_lower in _AUDIO_UNVERIFIED_DATASETS:
        if not audio_audited:
            msg = (
                f"Dataset '{target_dataset}' audio labels are NOT verified for deepfake claims. "
                "Restricting to visual-only transfer evaluation per PROJECT_BRIEF Section 9.2."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            checks["audio_labels_verified"] = False
            checks["audio_claim_allowed"] = False
        else:
            checks["audio_labels_verified"] = True
            checks["audio_claim_allowed"] = True
    else:
        checks["audio_labels_verified"] = target_lower in _AUDIO_VERIFIED_DATASETS
        checks["audio_claim_allowed"] = checks["audio_labels_verified"]

    return checks


def evaluate_cross_dataset(
    targets: np.ndarray,
    probabilities: np.ndarray,
    source_dataset: str,
    target_dataset: str,
    threshold: float | None = None,
    audio_audited: bool = False,
) -> CrossDatasetResult:
    """Evaluate cross-dataset transfer performance.

    AUROC is reported with NO threshold retuning (Section 9.2).
    A pre-selected threshold from validation is required.
    If the target dataset's audio labels are unverified (e.g., DFDC),
    only visual-transfer scope is allowed.

    Parameters
    ----------
    targets:
        Binary ground truth on target dataset ``(N,)``.
    probabilities:
        Predicted probabilities ``(N,)``.
    source_dataset:
        Name of the training dataset.
    target_dataset:
        Name of the evaluation dataset.
    threshold:
        Decision threshold pre-selected from source validation (no retuning).
    audio_audited:
        Whether audio labels in target dataset are audited.

    Returns
    -------
    CrossDatasetResult
        Transfer evaluation results with compatibility checks.
    """
    targets = np.asarray(targets).flatten()
    probabilities = np.asarray(probabilities).flatten()

    if len(targets) != len(probabilities):
        raise ValueError(
            f"Targets length ({len(targets)}) does not match probabilities length ({len(probabilities)})"
        )
    if len(targets) == 0:
        raise ValueError("Cannot evaluate cross-dataset metrics on empty data.")

    if threshold is None:
        raise ValueError(
            "Cross-dataset evaluation requires a pre-selected threshold from validation data. "
            "Auto-retuning threshold on test data is prohibited per PROJECT_BRIEF Section 9.2."
        )

    compatibility = check_label_compatibility(
        source_dataset, target_dataset, audio_audited=audio_audited
    )

    # Determine transfer type
    if compatibility.get("audio_claim_allowed", False):
        transfer_type = "full_av"
    else:
        transfer_type = "visual_only"

    # Compute metrics (no threshold retuning)
    metrics = compute_discrimination_metrics(
        targets, probabilities, threshold=threshold
    )

    result = CrossDatasetResult(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        transfer_type=transfer_type,
        metrics=metrics.to_dict(),
        compatibility_checks=compatibility,
    )

    logger.info(
        "Cross-dataset: %s → %s (type=%s) AUROC=%.4f",
        source_dataset, target_dataset, transfer_type,
        metrics.roc_auc,
    )

    return result
