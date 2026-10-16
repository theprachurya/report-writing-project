"""Measures parameter count, checkpoint size, peak GPU memory, and p50/p95 latency on named hardware.

Status: implemented — provides profile_model() for comprehensive efficiency profiling
including parameter count, checkpoint size, peak GPU memory, and latency percentiles.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyProfile:
    """Efficiency profiling results per Section 9.3.

    Attributes
    ----------
    total_params:
        Total number of parameters.
    trainable_params:
        Number of trainable parameters.
    total_params_m:
        Total parameters in millions.
    checkpoint_size_mb:
        Checkpoint file size in MB.
    peak_gpu_memory_mb:
        Peak GPU memory in MB during inference.
    latency_p50_ms:
        Median per-window inference latency in ms.
    latency_p95_ms:
        95th percentile latency in ms.
    throughput_windows_per_s:
        Windows processed per second.
    preprocessing_latency_p50_ms:
        Median per-window preprocessing latency in ms.
    preprocessing_latency_p95_ms:
        95th percentile preprocessing latency in ms.
    end_to_end_latency_p50_ms:
        Median end-to-end latency (preprocessing + inference) in ms.
    end_to_end_latency_p95_ms:
        95th percentile end-to-end latency in ms.
    hardware_name:
        Name of the hardware tested on.
    includes_preprocessing:
        Whether preprocessing time is included / measured.
    """
    total_params: int = 0
    trainable_params: int = 0
    total_params_m: float = 0.0
    checkpoint_size_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    throughput_windows_per_s: float = 0.0
    preprocessing_latency_p50_ms: float = 0.0
    preprocessing_latency_p95_ms: float = 0.0
    end_to_end_latency_p50_ms: float = 0.0
    end_to_end_latency_p95_ms: float = 0.0
    hardware_name: str = "unknown"
    includes_preprocessing: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging."""
        d: dict[str, Any] = {
            "total_params": self.total_params,
            "total_params_M": self.total_params_m,
            "trainable_params": self.trainable_params,
            "checkpoint_size_MB": self.checkpoint_size_mb,
            "peak_gpu_memory_MB": self.peak_gpu_memory_mb,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "throughput_windows/s": self.throughput_windows_per_s,
            "hardware": self.hardware_name,
            "includes_preprocessing": self.includes_preprocessing,
        }
        if self.includes_preprocessing or self.preprocessing_latency_p50_ms > 0:
            d["preprocessing_latency_p50_ms"] = self.preprocessing_latency_p50_ms
            d["preprocessing_latency_p95_ms"] = self.preprocessing_latency_p95_ms
            d["end_to_end_latency_p50_ms"] = self.end_to_end_latency_p50_ms
            d["end_to_end_latency_p95_ms"] = self.end_to_end_latency_p95_ms
        return d


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters.

    Parameters
    ----------
    model:
        PyTorch model.

    Returns
    -------
    tuple[int, int]
        ``(total_params, trainable_params)``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_checkpoint_size(
    model: nn.Module,
    save_path: str | Path | None = None,
) -> float:
    """Measure the checkpoint file size.

    Parameters
    ----------
    model:
        PyTorch model.
    save_path:
        Path to save a temporary checkpoint.

    Returns
    -------
    float
        Checkpoint size in MB.
    """
    import tempfile

    if save_path is None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_path = f.name

    save_path = Path(save_path)
    torch.save(model.state_dict(), str(save_path))
    size_bytes = save_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Clean up temp file
    if save_path.name.startswith("tmp"):
        save_path.unlink(missing_ok=True)

    return size_mb


def measure_latency(
    model: nn.Module,
    sample_input: dict[str, torch.Tensor],
    n_warmup: int = 10,
    n_measure: int = 100,
    device: torch.device | str = "cpu",
) -> tuple[float, float, float]:
    """Measure per-window inference latency.

    Parameters
    ----------
    model:
        PyTorch model in eval mode.
    sample_input:
        Example input dict.
    n_warmup:
        Warmup iterations (not measured).
    n_measure:
        Measurement iterations.
    device:
        Target device.

    Returns
    -------
    tuple[float, float, float]
        ``(p50_ms, p95_ms, throughput_windows_per_s)``.
    """
    import numpy as np

    model = model.to(device)
    model.eval()

    # Move input to device
    device_input = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample_input.items()
    }

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**device_input)

    # Synchronize if GPU
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(n_measure):
            start = time.perf_counter()
            model(**device_input)
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    # Report p50 and p95 latency percentiles per Section 9.3
    percentiles = np.percentile(latencies, [50, 95])
    p50 = float(percentiles[0])
    p95 = float(percentiles[1])
    throughput = 1000.0 / np.mean(latencies) if len(latencies) > 0 and np.mean(latencies) > 0 else 0.0

    return p50, p95, throughput


def measure_preprocessing_latency(
    preprocess_fn: Any,
    sample_raw_input: Any,
    n_warmup: int = 5,
    n_measure: int = 50,
) -> tuple[float, float]:
    """Measure per-window preprocessing latency separately.

    Parameters
    ----------
    preprocess_fn:
        Preprocessing function that transforms raw input to model input.
    sample_raw_input:
        Example raw input clip/data.
    n_warmup:
        Warmup iterations.
    n_measure:
        Measurement iterations.

    Returns
    -------
    tuple[float, float]
        ``(p50_ms, p95_ms)``.
    """
    import numpy as np

    for _ in range(n_warmup):
        preprocess_fn(sample_raw_input)

    latencies: list[float] = []
    for _ in range(n_measure):
        start = time.perf_counter()
        preprocess_fn(sample_raw_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    percentiles = np.percentile(latencies, [50, 95])
    return float(percentiles[0]), float(percentiles[1])


def profile_model(
    model: nn.Module,
    sample_input: dict[str, torch.Tensor] | None = None,
    device: str = "cpu",
    hardware_name: str | None = None,
    n_measure: int = 100,
    preprocess_fn: Any | None = None,
    sample_raw_input: Any | None = None,
) -> EfficiencyProfile:
    """Run full efficiency profiling on a model.

    Parameters
    ----------
    model:
        PyTorch model to profile.
    sample_input:
        Example input for latency measurement.
    device:
        Target device.
    hardware_name:
        Human-readable hardware name.
    n_measure:
        Number of latency measurement iterations.
    preprocess_fn:
        Optional preprocessing function to benchmark preprocessing separately.
    sample_raw_input:
        Raw input for preprocessing latency benchmark.

    Returns
    -------
    EfficiencyProfile
        Complete efficiency profile.
    """
    total_params, trainable_params = count_parameters(model)
    ckpt_size = measure_checkpoint_size(model)

    profile = EfficiencyProfile(
        total_params=total_params,
        trainable_params=trainable_params,
        total_params_m=total_params / 1e6,
        checkpoint_size_mb=ckpt_size,
        hardware_name=hardware_name or device,
    )

    # Peak GPU memory profiling via torch.cuda.max_memory_allocated()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.to(device)
        if sample_input:
            with torch.no_grad():
                device_input = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_input.items()
                }
                model(**device_input)
            profile.peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    # Preprocessing latency profiling
    if preprocess_fn is not None and sample_raw_input is not None:
        prep_p50, prep_p95 = measure_preprocessing_latency(
            preprocess_fn, sample_raw_input, n_measure=min(n_measure, 50)
        )
        profile.preprocessing_latency_p50_ms = prep_p50
        profile.preprocessing_latency_p95_ms = prep_p95
        profile.includes_preprocessing = True

    # Inference latency profiling
    if sample_input is not None:
        p50, p95, throughput = measure_latency(
            model, sample_input, n_measure=n_measure, device=device
        )
        profile.latency_p50_ms = p50
        profile.latency_p95_ms = p95
        profile.throughput_windows_per_s = throughput

        if profile.includes_preprocessing:
            profile.end_to_end_latency_p50_ms = profile.preprocessing_latency_p50_ms + p50
            profile.end_to_end_latency_p95_ms = profile.preprocessing_latency_p95_ms + p95

    logger.info(
        "Efficiency: %.2fM params, %.1fMB checkpoint, peak GPU=%.1fMB, p50=%.1fms, p95=%.1fms",
        profile.total_params_m,
        profile.checkpoint_size_mb,
        profile.peak_gpu_memory_mb,
        profile.latency_p50_ms,
        profile.latency_p95_ms,
    )

    return profile
