"""Deterministic seeding helpers for reproducible runs across three final seeds.

Status: implemented — provides set_seed() for global reproducibility and
SeedContext for scoped seeding. Seeds are logged with every call.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + all CUDA devices).
    Optionally forces deterministic cuDNN behaviour for bitwise
    reproducibility at the cost of potential slowdown.

    Parameters
    ----------
    seed:
        The integer seed to use across all RNG backends.
    deterministic_cudnn:
        If ``True`` (default), set ``torch.backends.cudnn.deterministic = True``
        and ``torch.backends.cudnn.benchmark = False``.  Reproducibility
        checklist (Section 13) requires seeds and determinism settings to be
        logged.

    Notes
    -----
    Call this function at the very start of every training / evaluation script
    and log the seed alongside the run configuration (see
    ``logging_utils.log_config``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    # PyTorch ≥ 1.8 environment variable for full determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        # Older PyTorch versions or operations without deterministic implementations
        pass

    logger.info(
        "Seed set to %d (deterministic_cudnn=%s)",
        seed,
        deterministic_cudnn,
    )


def get_seed_from_config(seeds: list[int], run_index: int = 0) -> int:
    """Select a seed from the list of three final seeds.

    Parameters
    ----------
    seeds:
        List of seeds from ``train_config.yaml`` (e.g. ``[42, 123, 7]``).
    run_index:
        Which run (0-indexed) to select.  Must be < ``len(seeds)``.

    Returns
    -------
    int
        The selected seed.

    Raises
    ------
    IndexError
        If *run_index* is out of range.
    """
    if run_index < 0 or run_index >= len(seeds):
        raise IndexError(
            f"run_index={run_index} out of range for {len(seeds)} seeds"
        )
    seed = seeds[run_index]
    logger.info("Selected seed %d for run index %d", seed, run_index)
    return seed


class SeedContext:
    """Context manager that sets a seed on entry and restores RNG states on exit.

    Useful for ensuring deterministic preprocessing or evaluation while
    leaving the outer training loop's RNG state undisturbed.

    Example
    -------
    >>> with SeedContext(99):
    ...     x = torch.randn(3)  # deterministic
    >>> y = torch.randn(3)      # continues from prior state
    """

    def __init__(self, seed: int, *, deterministic_cudnn: bool = False) -> None:
        self.seed = seed
        self.deterministic_cudnn = deterministic_cudnn
        self._python_state: tuple[Any, ...] | None = None
        self._numpy_state: tuple[Any, ...] | dict[str, Any] | None = None
        self._torch_state: torch.Tensor | None = None
        self._cuda_states: list[torch.Tensor] | None = None
        self._cudnn_deterministic: bool | None = None
        self._cudnn_benchmark: bool | None = None
        self._cuda_matmul_allow_tf32: bool | None = None
        self._cudnn_allow_tf32: bool | None = None

    def __enter__(self) -> "SeedContext":
        # Save current states
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self._cuda_states = [
                torch.cuda.get_rng_state(i)
                for i in range(torch.cuda.device_count())
            ]
        self._cudnn_deterministic = torch.backends.cudnn.deterministic
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            self._cuda_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            self._cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

        # Apply seed
        set_seed(self.seed, deterministic_cudnn=self.deterministic_cudnn)
        return self

    def __exit__(self, *exc: object) -> None:
        # Restore states with safety guards
        if self._python_state is not None:
            random.setstate(self._python_state)
        if self._numpy_state is not None:
            np.random.set_state(self._numpy_state)
        if self._torch_state is not None:
            torch.random.set_rng_state(self._torch_state)
        if self._cuda_states is not None and torch.cuda.is_available():
            for i, state in enumerate(self._cuda_states):
                torch.cuda.set_rng_state(state, i)
        if self._cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = self._cudnn_deterministic
        if self._cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = self._cudnn_benchmark
        if (
            self._cuda_matmul_allow_tf32 is not None
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "matmul")
        ):
            torch.backends.cuda.matmul.allow_tf32 = self._cuda_matmul_allow_tf32
        if (
            self._cudnn_allow_tf32 is not None
            and hasattr(torch.backends, "cudnn")
            and hasattr(torch.backends.cudnn, "allow_tf32")
        ):
            torch.backends.cudnn.allow_tf32 = self._cudnn_allow_tf32
