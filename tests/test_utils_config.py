"""Unit tests for Utils & Config subsystem.

Tests config_loader, seed_utils, logging_utils, manifest_builder, and hashing fixes.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.config_loader import (
    DataConfig,
    ModelConfig,
    _build_dataclass,
    _expand_env_vars,
    load_config,
    load_data_config,
    load_eval_config,
    load_model_config,
    load_train_config,
    load_yaml,
    save_config_snapshot,
)
from src.utils.hashing import (
    check_cross_partition_duplicates,
    compute_file_hash,
    hamming_distance,
)
from src.utils.logging_utils import (
    ExperimentLogger,
    LoggingBackend,
    save_config_with_checkpoint,
    setup_logging,
)
from src.utils.manifest_builder import SOURCE_TABLE_COLUMNS, validate_source_table
from src.utils.seed_utils import SeedContext, set_seed


# ── Config Loader Tests ───────────────────────────────────────────────────────

def test_load_yaml_non_dict_raises_value_error(tmp_path: Path) -> None:
    """Non-dict YAML content raises a clear ValueError."""
    list_yaml = tmp_path / "list_config.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping \\(dict\\)"):
        load_yaml(list_yaml)

    scalar_yaml = tmp_path / "scalar_config.yaml"
    scalar_yaml.write_text("hello world\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping \\(dict\\)"):
        load_yaml(scalar_yaml)


def test_env_var_expansion_with_fallback() -> None:
    """Env var expansion supports ${VAR:-default} fallback pattern."""
    # When unset, uses default
    os.environ.pop("TEST_DATASET_ROOT", None)
    expanded = _expand_env_vars("${TEST_DATASET_ROOT:-data/default_path}")
    assert expanded == "data/default_path"

    # When set, uses env value
    os.environ["TEST_DATASET_ROOT"] = "/custom/dataset/path"
    try:
        expanded = _expand_env_vars("${TEST_DATASET_ROOT:-data/default_path}")
        assert expanded == "/custom/dataset/path"
    finally:
        os.environ.pop("TEST_DATASET_ROOT", None)


def test_save_config_snapshot_includes_hash(tmp_path: Path) -> None:
    """save_config_snapshot saves YAML with sha256 header and fields."""
    cfg = {"stage": "unimodal", "loss": {"tau_r": 1.0}}
    snap_path = save_config_snapshot(cfg, tmp_path / "test_snapshot.yaml")
    assert snap_path.exists()

    content = snap_path.read_text(encoding="utf-8")
    assert "# SHA256:" in content

    loaded = yaml.safe_load(content)
    assert "config_sha256" in loaded
    assert len(loaded["config_sha256"]) == 64
    assert loaded["stage"] == "unimodal"


@dataclass
class _SubTestConfig:
    value: int = 10


@dataclass
class _RootTestConfig:
    sub: _SubTestConfig = dataclasses.field(default_factory=_SubTestConfig)
    name: str = "test"


def test_build_dataclass_with_future_annotations() -> None:
    """Nested dataclasses resolve correctly without eval using get_type_hints."""
    built = _build_dataclass(_RootTestConfig, {"sub": {"value": 42}, "name": "custom"})
    assert isinstance(built, _RootTestConfig)
    assert isinstance(built.sub, _SubTestConfig)
    assert built.sub.value == 42
    assert built.name == "custom"


def test_build_dataclass_missing_required_field_logs_error() -> None:
    """Missing required positional field logs error and raises TypeError."""
    @dataclass
    class StrictConfig:
        required_val: int

    with pytest.raises(TypeError):
        _build_dataclass(StrictConfig, {})


def test_config_path_resolution(tmp_path: Path) -> None:
    """load_*_config correctly resolves config_path parent dir."""
    custom_dir = tmp_path / "custom_configs"
    custom_dir.mkdir(parents=True)
    custom_file = custom_dir / "data_config.yaml"

    # Write minimal valid data config
    custom_file.write_text("decode:\n  target_fps: 30\n", encoding="utf-8")
    cfg = load_data_config(config_path=custom_file)
    assert cfg.decode.target_fps == 30


# ── Seed Utils Tests ──────────────────────────────────────────────────────────

def test_set_seed_and_seed_context() -> None:
    """set_seed and SeedContext manage random state deterministically."""
    set_seed(42, deterministic_cudnn=True)
    val1 = np.random.rand()

    with SeedContext(999, deterministic_cudnn=True) as ctx:
        assert ctx._numpy_state is not None
        np_inner = np.random.rand()

    set_seed(42, deterministic_cudnn=True)
    val2 = np.random.rand()
    assert val1 == val2


def test_seed_context_safe_exit_with_none_states() -> None:
    """SeedContext.__exit__ handles uninitialized / None states safely."""
    ctx = SeedContext(42)
    # Exiting before entering should not crash
    ctx.__exit__(None, None, None)


# ── Logging Utils Tests ───────────────────────────────────────────────────────

def test_setup_logging_creates_parent_dir(tmp_path: Path) -> None:
    """setup_logging creates parent directory for log file if it does not exist."""
    nested_log = tmp_path / "nested" / "sub" / "run.log"
    setup_logging(log_file=nested_log)
    assert nested_log.parent.exists()


def test_experiment_logger_fault_tolerance() -> None:
    """Failing backends do not crash ExperimentLogger methods."""
    class FailingBackend(LoggingBackend):
        def log_params(self, params: dict[str, Any]) -> None:
            raise RuntimeError("Backend disk error")
        def log_metric(self, key: str, value: float, step: int | None = None) -> None:
            raise ConnectionError("Backend network error")
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            raise TimeoutError("Backend timeout")
        def log_artifact(self, local_path: str | Path) -> None:
            raise OSError("Backend write failed")
        def set_tag(self, key: str, value: str) -> None:
            raise ValueError("Invalid tag")
        def finish(self) -> None:
            raise RuntimeError("Backend finish error")

    exp_logger = ExperimentLogger(backends=[FailingBackend()])
    # None of these should raise
    exp_logger.log_params({"lr": 0.001})
    exp_logger.log_metric("loss", 0.5, step=1)
    exp_logger.log_metrics({"acc": 0.95}, step=1)
    exp_logger.log_artifact("non_existent_file.txt")
    exp_logger.set_tag("model", "test")
    exp_logger.finish()


def test_save_config_with_checkpoint_dataclass(tmp_path: Path) -> None:
    """save_config_with_checkpoint supports dataclass configs."""
    @dataclass
    class DummyConfig:
        learning_rate: float = 1e-3
        batch_size: int = 16

    saved = save_config_with_checkpoint(DummyConfig(), tmp_path / "checkpoints")
    assert saved.exists()
    content = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert content["learning_rate"] == 0.001
    assert content["batch_size"] == 16


# ── Manifest Builder Tests ────────────────────────────────────────────────────

def test_validate_source_table_boolean_representations() -> None:
    """validate_source_table safely accepts bool and string representations of bool."""
    records = []
    for i in range(5):
        records.append({
            "clip_id": f"clip_{i}",
            "dataset": "fakeavceleb",
            "source_identity": f"id_{i}",
            "source_video": f"video_{i}",
            "manipulation_family": "real" if i % 2 == 0 else "faceswap",
            "video_manipulated": i % 2 != 0,
            "audio_manipulated": False,
            "clip_fake": "True" if i % 2 != 0 else "False",  # string bools
            "audio_condition": "original",
            "file_path": f"path/to/{i}.mp4",
            "duration_s": 2.0,
            "file_hash": f"hash_{i}",
        })
    df = pd.DataFrame(records)
    # Should succeed without ValueError
    validate_source_table(df)

    # Boolean dtype should also succeed
    df["clip_fake"] = [i % 2 != 0 for i in range(5)]
    validate_source_table(df)

    # Integer 0/1 should also succeed
    df["clip_fake"] = [i % 2 for i in range(5)]
    validate_source_table(df)

    # Invalid string should raise ValueError
    df["clip_fake"] = df["clip_fake"].astype(object)
    df.at[0, "clip_fake"] = "maybe"
    with pytest.raises(ValueError, match="clip_fake must contain only binary labels"):
        validate_source_table(df)
