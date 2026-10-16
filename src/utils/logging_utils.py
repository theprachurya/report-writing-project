"""Thin wrapper around MLflow/TensorBoard for consistent experiment logging.

Status: implemented — provides ExperimentLogger facade with MLflow and
TensorBoard backends, plus helpers to save configs alongside checkpoints.
See AGENTS.md for build order.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ── Abstract backend ─────────────────────────────────────────────────────────

class LoggingBackend(ABC):
    """Abstract interface for experiment logging backends."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dict of hyperparameters / configuration values."""

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric."""

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log a dict of scalar metrics."""

    @abstractmethod
    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file as an artifact (config, figure, etc.)."""

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """Set an experiment tag / metadata entry."""

    @abstractmethod
    def finish(self) -> None:
        """Finalize the logging run (flush, close connections)."""


# ── MLflow backend ───────────────────────────────────────────────────────────

class MLflowBackend(LoggingBackend):
    """MLflow-backed experiment logger.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name.
    run_name:
        Optional name for this specific run.
    tracking_uri:
        MLflow tracking URI.  Defaults to a local ``mlruns/`` directory.
    """

    def __init__(
        self,
        experiment_name: str = "deepfake_detection",
        run_name: str | None = None,
        tracking_uri: str = "mlruns",
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is required for MLflowBackend. "
                "Install it with: pip install mlflow"
            ) from exc

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        logger.info(
            "MLflow run started: %s (experiment=%s)",
            self._run.info.run_id,
            experiment_name,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters, flattening nested dicts with dot notation."""
        flat = _flatten_dict(params)
        # MLflow limits param value length to 500 chars
        for k, v in flat.items():
            str_v = str(v)
            if len(str_v) > 500:
                str_v = str_v[:497] + "..."
            self._mlflow.log_param(k, str_v)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        self._mlflow.log_artifact(str(local_path))

    def set_tag(self, key: str, value: str) -> None:
        self._mlflow.set_tag(key, value)

    def finish(self) -> None:
        self._mlflow.end_run()
        logger.info("MLflow run ended.")


# ── TensorBoard backend ─────────────────────────────────────────────────────

class TensorBoardBackend(LoggingBackend):
    """TensorBoard-backed experiment logger.

    Parameters
    ----------
    log_dir:
        Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str = "runs") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard is required for TensorBoardBackend. "
                "Install it with: pip install tensorboard"
            ) from exc

        self._writer = SummaryWriter(log_dir=log_dir)
        self._log_dir = Path(log_dir)
        logger.info("TensorBoard writer initialized at %s", log_dir)

    def log_params(self, params: dict[str, Any]) -> None:
        """Write parameters as a text summary in TensorBoard."""
        flat = _flatten_dict(params)
        text = "\n".join(f"**{k}**: {v}" for k, v in sorted(flat.items()))
        self._writer.add_text("hyperparameters", text)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._writer.add_scalar(key, value, global_step=step)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        """Copy artifact to the TensorBoard log directory for preservation."""
        src = Path(local_path)
        if not src.exists():
            logger.warning("Artifact not found, skipping copy: %s", src)
            return
        dst = self._log_dir / "artifacts" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("Artifact copied to %s", dst)

    def set_tag(self, key: str, value: str) -> None:
        self._writer.add_text(f"tags/{key}", value)

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()
        logger.info("TensorBoard writer closed.")


# ── Unified facade ───────────────────────────────────────────────────────────

class ExperimentLogger:
    """Unified experiment-logging facade that delegates to one or more backends.

    Parameters
    ----------
    backends:
        List of :class:`LoggingBackend` instances.  If empty, a console-only
        fallback is used (metrics printed via ``logging``).

    Example
    -------
    >>> exp_logger = ExperimentLogger.from_config({"backend": "tensorboard"})
    >>> exp_logger.log_metrics({"val_auroc": 0.87, "val_loss": 0.42}, step=5)
    >>> exp_logger.finish()
    """

    def __init__(self, backends: list[LoggingBackend] | None = None) -> None:
        self._backends = backends or []
        if not self._backends:
            logger.warning(
                "No logging backends configured — metrics will only appear in "
                "the console log."
            )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        experiment_name: str = "deepfake_detection",
        run_name: str | None = None,
    ) -> "ExperimentLogger":
        """Create an ExperimentLogger from a logging config dict.

        Supported ``config["backend"]`` values:
        ``"mlflow"``, ``"tensorboard"``, ``"both"``, ``"none"``.
        """
        backend_name = config.get("backend", "tensorboard").lower()
        backends: list[LoggingBackend] = []

        if backend_name in ("mlflow", "both"):
            backends.append(
                MLflowBackend(
                    experiment_name=experiment_name,
                    run_name=run_name,
                    tracking_uri=config.get("mlflow_tracking_uri", "mlruns"),
                )
            )
        if backend_name in ("tensorboard", "both"):
            backends.append(
                TensorBoardBackend(
                    log_dir=config.get("tensorboard_log_dir", "runs"),
                )
            )
        return cls(backends=backends)

    # ── Delegation ───────────────────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters / configuration to all backends."""
        for b in self._backends:
            try:
                b.log_params(params)
            except Exception as exc:
                logger.warning("Backend %s failed to log_params: %s", b, exc)
        logger.debug("Logged %d params", len(_flatten_dict(params)))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric to all backends."""
        for b in self._backends:
            try:
                b.log_metric(key, value, step=step)
            except Exception as exc:
                logger.warning("Backend %s failed to log_metric: %s", b, exc)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log a dict of scalar metrics to all backends."""
        for b in self._backends:
            try:
                b.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.warning("Backend %s failed to log_metrics: %s", b, exc)
        logger.info("Step %s metrics: %s", step, metrics)

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file artifact to all backends."""
        for b in self._backends:
            try:
                b.log_artifact(local_path)
            except Exception as exc:
                logger.warning("Backend %s failed to log_artifact: %s", b, exc)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag/metadata entry on all backends."""
        for b in self._backends:
            try:
                b.set_tag(key, value)
            except Exception as exc:
                logger.warning("Backend %s failed to set_tag: %s", b, exc)

    def finish(self) -> None:
        """Finalize all backends."""
        for b in self._backends:
            try:
                b.finish()
            except Exception as exc:
                logger.warning("Backend %s failed to finish: %s", b, exc)


# ── Config saving (reproducibility checklist) ────────────────────────────────

def save_config_with_checkpoint(
    config: dict[str, Any] | Any,
    checkpoint_dir: str | Path,
    config_name: str = "config.yaml",
) -> Path:
    """Save a configuration dict alongside a model checkpoint.

    This satisfies the reproducibility checklist item: *"Config file saved
    with every checkpoint and metric artifact."*

    Parameters
    ----------
    config:
        The full merged config dict or dataclass instance to persist.
    checkpoint_dir:
        Directory where the checkpoint is saved.
    config_name:
        Filename for the saved config.

    Returns
    -------
    Path
        Path to the saved config file.
    """
    if dataclasses.is_dataclass(config):
        cfg_dict = dataclasses.asdict(config)
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = dict(config)

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config_name
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg_dict, fh, default_flow_style=False, sort_keys=False)
    logger.info("Config saved to %s", out_path)
    return out_path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> None:
    """Configure the root logger for consistent formatting.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.INFO``).
    log_file:
        Optional path to a log file; if provided, a ``FileHandler`` is added.
    fmt:
        Log message format string.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logger.info("Logging configured: level=%s, file=%s", level, log_file)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
