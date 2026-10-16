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
        """Generate all degradation conditions to evaluate per Section 9.2.

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
        # Frame drop per Section 9.2 (frame_drop_rates: [0.1, 0.2, 0.3])
        frame_drop_rates = (
            self.video_cfg.get("frame_drop_rates")
            or self.video_cfg.get("frame_drop_rate")
            or [0.1, 0.2, 0.3]
        )
        if isinstance(frame_drop_rates, (int, float)):
            frame_drop_rates = [frame_drop_rates]
        for rate in frame_drop_rates:
            conditions.append({
                "type": "video", "name": f"frame_drop_{rate}",
                "params": {"frame_drop_rate": rate},
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
        # Packet dropout / segment dropout per Section 9.2
        dropout_rates = (
            self.audio_cfg.get("dropout_rates")
            or self.audio_cfg.get("dropout_rate")
            or [0.05, 0.10, 0.20]
        )
        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [dropout_rates]
        for rate in dropout_rates:
            conditions.append({
                "type": "audio", "name": f"packet_dropout_{rate}",
                "params": {"dropout_rate": rate},
            })
        # Codec re-sampling per Section 9.2
        resample_rates = self.audio_cfg.get("resample_rates", [8000, 16000, 22050])
        if isinstance(resample_rates, (int, float)):
            resample_rates = [resample_rates]
        for sr in resample_rates:
            conditions.append({
                "type": "audio", "name": f"codec_resample_{sr}Hz",
                "params": {"sample_rate": sr},
            })

        # Joint degradation
        if self.joint_enabled:
            conditions.append({
                "type": "joint", "name": "joint_blur_and_noise",
                "params": {"blur_sigma": 2.0, "snr_db": 10},
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
        # Safe division guard in relative_change
        if clean_metric is None or np.isnan(clean_metric) or abs(clean_metric) < 1e-12:
            rel_change = 0.0
        else:
            rel_change = float(change / clean_metric * 100.0)

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

    def run_all(
        self,
        evaluate_fn: Callable,
        clean_metric: float,
        clean_w_v: float = 0.5,
        conditions: list[dict[str, Any]] | None = None,
    ) -> DegradationSuiteResults:
        """Run all degradation conditions and collect aggregate results.

        Parameters
        ----------
        evaluate_fn:
            Function that accepts condition params and returns
            ``(metric, mean_w_v, mean_w_a)``.
        clean_metric:
            Metric value on clean data for comparison.
        clean_w_v:
            Clean visual gate weight for shift calculation.
        conditions:
            Optional list of conditions to run. If None, uses get_degradation_conditions().

        Returns
        -------
        DegradationSuiteResults
            Aggregate results from the full degradation battery.
        """
        if conditions is None:
            conditions = self.get_degradation_conditions()

        results = [
            self.run_condition(cond, evaluate_fn, clean_metric, clean_w_v)
            for cond in conditions
        ]

        summary = {
            "n_conditions": len(results),
            "mean_change": float(np.mean([r.metric_change for r in results])) if results else 0.0,
            "mean_rel_change": float(np.mean([r.relative_change for r in results])) if results else 0.0,
        }
        return DegradationSuiteResults(results=results, summary=summary)


# Alias for canonical naming
DegradationSuite = DegradationSuiteRunner
