"""Runs the video/audio/joint degradation battery (H.264 CRF 18/28/38, resize, blur, AWGN SNR levels, dropout) — Section 9.2.

Status: implemented — provides DegradationSuiteRunner that systematically applies
controlled degradations and measures performance change and gate shift.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DegradationResult:
    """Result from a single degradation condition.

    Attributes
    ----------
    degradation_type:
        Category: ``"video"``, ``"audio"``, ``"joint"``, ``"missing"``.
    degradation_name:
        Specific degradation applied.
    parameters:
        Degradation parameters.
    clean_metric:
        Metric value on clean data.
    degraded_metric:
        Metric value after degradation.
    metric_change:
        Absolute change from clean.
    relative_change:
        Relative change from clean (%).
    mean_w_v:
        Mean visual gate weight under this degradation.
    mean_w_a:
        Mean audio gate weight under this degradation.
    gate_shift:
        Change in gate weights vs clean condition.
    """
    degradation_type: str
    degradation_name: str
    parameters: dict[str, Any]
    clean_metric: float = 0.0
    degraded_metric: float = 0.0
    metric_change: float = 0.0
    relative_change: float = 0.0
    mean_w_v: float = 0.5
    mean_w_a: float = 0.5
    gate_shift: float = 0.0


@dataclass
class DegradationSuiteResults:
    """Aggregate results from the full degradation suite.

    Attributes
    ----------
    results:
        List of per-condition results.
    summary:
        Summary statistics.
    """
    results: list[DegradationResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_table(self) -> list[dict[str, Any]]:
        """Convert results to a table-friendly format."""
        return [
            {
                "type": r.degradation_type,
                "name": r.degradation_name,
                "params": str(r.parameters),
                "clean": f"{r.clean_metric:.4f}",
                "degraded": f"{r.degraded_metric:.4f}",
                "change": f"{r.metric_change:+.4f}",
                "relative_%": f"{r.relative_change:+.1f}",
                "w_v": f"{r.mean_w_v:.3f}",
                "w_a": f"{r.mean_w_a:.3f}",
                "gate_shift": f"{r.gate_shift:+.3f}",
            }
            for r in self.results
        ]


class DegradationSuiteRunner:
    """Runs the controlled degradation battery per Section 9.2.

    Parameters
    ----------
    eval_config:
        Evaluation configuration with degradation parameters.
    """

    def __init__(self, eval_config: Any = None) -> None:
        if eval_config is None:
            from src.utils.config_loader import load_eval_config
            eval_config = load_eval_config()

        cfg = eval_config
        deg = cfg.degradation if hasattr(cfg, 'degradation') else cfg.get("degradation", {})

        if isinstance(deg, dict):
            self.video_cfg = deg.get("video", {})
            self.audio_cfg = deg.get("audio", {})
            self.joint_enabled = deg.get("joint", {}).get("enabled", True)
            self.missing_cfg = deg.get("missing_modality", {})
        else:
            self.video_cfg = {}
            self.audio_cfg = {}
            self.joint_enabled = True
            self.missing_cfg = {}

    def get_degradation_conditions(self) -> list[dict[str, Any]]:
        """Generate all degradation conditions to evaluate.

        Returns
        -------
        list[dict[str, Any]]
            List of condition dicts with type, name, and parameters.
        """
        conditions = []

        # Video degradations
        for crf in self.video_cfg.get("crf_levels", [18, 28, 38]):
            conditions.append({
                "type": "video", "name": f"h264_crf_{crf}",
                "params": {"crf": crf},
            })
        for res in self.video_cfg.get("resize_resolutions", [720, 480, 360]):
            conditions.append({
                "type": "video", "name": f"resize_{res}p",
                "params": {"resolution": res},
            })
        for sigma in self.video_cfg.get("blur_sigma", [1.0, 2.0, 3.0]):
            conditions.append({
                "type": "video", "name": f"blur_sigma_{sigma}",
                "params": {"sigma": sigma},
            })

        # Audio degradations
        for snr in self.audio_cfg.get("awgn_snr_db", [20, 10, 0]):
            conditions.append({
                "type": "audio", "name": f"awgn_{snr}dB",
                "params": {"snr_db": snr},
            })
        for cutoff in self.audio_cfg.get("lowpass_cutoff_hz", [4000, 2000, 1000]):
            conditions.append({
                "type": "audio", "name": f"lowpass_{cutoff}Hz",
                "params": {"cutoff_hz": cutoff},
            })

        # Missing modality
        conditions.append({
            "type": "missing", "name": "zeroed_audio",
            "params": {"modality": "audio"},
        })
        conditions.append({
            "type": "missing", "name": "no_face",
            "params": {"modality": "visual"},
        })

        return conditions

    def run_condition(
        self,
        condition: dict[str, Any],
        evaluate_fn: Callable,
        clean_metric: float,
        clean_w_v: float = 0.5,
    ) -> DegradationResult:
        """Evaluate a single degradation condition.

        Parameters
        ----------
        condition:
            Degradation condition dict.
        evaluate_fn:
            Function that accepts condition params and returns
            ``(metric, mean_w_v, mean_w_a)``.
        clean_metric:
            Metric value on clean data for comparison.
        clean_w_v:
            Clean visual gate weight for shift calculation.

        Returns
        -------
        DegradationResult
            Result for this condition.
        """
        metric, mean_w_v, mean_w_a = evaluate_fn(condition)

        change = metric - clean_metric
        rel_change = (change / clean_metric * 100) if clean_metric != 0 else 0

        return DegradationResult(
            degradation_type=condition["type"],
            degradation_name=condition["name"],
            parameters=condition["params"],
            clean_metric=clean_metric,
            degraded_metric=metric,
            metric_change=change,
            relative_change=rel_change,
            mean_w_v=mean_w_v,
            mean_w_a=mean_w_a,
            gate_shift=mean_w_v - clean_w_v,
        )
