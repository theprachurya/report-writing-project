"""Evaluation subpackage for metrics, calibration, degradation testing, and profiling.

See AGENTS.md for build order.
"""

from src.evaluate.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    compute_calibration_metrics,
    plot_reliability_diagram,
)
from src.evaluate.metrics import (
    DiscriminationMetrics,
    bootstrap_confidence_interval,
    compute_discrimination_metrics,
    find_optimal_threshold,
)
from src.evaluate.mlaad_audio import (
    AudioEvalRecord,
    MlaadAudioDataset,
    bootstrap_grouped_auc,
    build_mlaad_audio_records,
    predict_audio_head,
)

__all__ = [
    "CalibrationMetrics",
    "DiscriminationMetrics",
    "AudioEvalRecord",
    "MlaadAudioDataset",
    "TemperatureScaler",
    "bootstrap_confidence_interval",
    "bootstrap_grouped_auc",
    "build_mlaad_audio_records",
    "compute_calibration_metrics",
    "compute_discrimination_metrics",
    "find_optimal_threshold",
    "plot_reliability_diagram",
    "predict_audio_head",
]
