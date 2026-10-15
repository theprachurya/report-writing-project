"""Loads and validates the YAML files under configs/ into typed config objects.

Status: implemented — provides load_config(), load_data_config(), load_model_config(),
load_train_config(), and load_eval_config() with environment variable expansion and
merge-override support. See AGENTS.md for build order.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Environment variable expansion ───────────────────────────────────────────

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` patterns in string values using ``os.environ``.

    If the environment variable is not set the raw ``${VAR}`` string is kept so
    that code can detect and raise an informative error instead of silently
    using a partial path.
    """
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


# ── Raw YAML loading ─────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return the parsed dictionary.

    Environment variables in ``${VAR}`` format are expanded after parsing.

    Parameters
    ----------
    path:
        Absolute or project-relative path to a ``.yaml`` config file.

    Returns
    -------
    dict[str, Any]
        The parsed and env-expanded config dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return _expand_env_vars(raw)


def merge_configs(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *overrides* into *base*, returning a new dict.

    - Nested dicts are merged recursively.
    - Lists and scalars in *overrides* replace their counterparts in *base*.

    Parameters
    ----------
    base:
        Base configuration dictionary.
    overrides:
        Override dictionary whose values take precedence.

    Returns
    -------
    dict[str, Any]
        Merged configuration.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Typed config dataclasses ─────────────────────────────────────────────────

@dataclass
class DecodeConfig:
    """Video decode parameters (Section 6)."""
    target_fps: int = 25
    window_duration_s: float = 2.0
    frames_per_window: int = 16
    effective_visual_fps: int = 8


@dataclass
class FaceConfig:
    """Face extraction parameters (Section 6)."""
    detector: str = "retinaface"
    box_expansion: float = 0.20
    crop_size: int = 160
    min_valid_frames: int = 12
    align_by_landmarks: bool = True


@dataclass
class VisualNormalizationConfig:
    """RGB normalization used by the visual encoder."""
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataQualityConfig:
    """Feature-vector widths recorded in the local feature cache."""
    visual_dim: int = 7
    audio_dim: int = 5


@dataclass
class AudioConfig:
    """Audio extraction parameters (Section 6)."""
    sample_rate: int = 16_000
    channels: int = 1
    peak_normalize: bool = True
    vad_threshold: float = 0.30


@dataclass
class MelConfig:
    """Log-Mel spectrogram parameters (Section 6)."""
    n_mels: int = 80
    window_length_ms: int = 25
    hop_length_ms: int = 10
    f_min: float = 20.0
    f_max: float = 8000.0


@dataclass
class AlignmentConfig:
    """Temporal alignment parameters (Section 6)."""
    temporal_bins: int = 8
    embedding_dim: int = 256


@dataclass
class ValidityConfig:
    """Window validity thresholds (Section 6)."""
    min_face_detection_rate: float = 0.75
    min_vad_fraction: float = 0.10
    retain_invalid_with_flags: bool = True


@dataclass
class SplitConfig:
    """Dataset split ratios (Section 5.3)."""
    train: float = 0.70
    val: float = 0.10
    test: float = 0.20
    seed: int = 42


@dataclass
class CacheConfig:
    """Feature caching paths."""
    cache_dir: str = "data/cache"
    enable_crop_cache: bool = True
    enable_mel_cache: bool = True


@dataclass
class DataConfig:
    """Top-level data configuration aggregating all sub-configs."""
    datasets: dict[str, Any] = field(default_factory=dict)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    visual_normalization: VisualNormalizationConfig = field(default_factory=VisualNormalizationConfig)
    quality_features: DataQualityConfig = field(default_factory=DataQualityConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class VisualEncoderConfig:
    """Visual encoder architecture (Section 7.1)."""
    backbone: str = "mobilenetv3_small"
    pretrained: bool = True
    input_size: int = 160
    temporal_bins: int = 8
    embedding_dim: int = 256
    temporal_conv: dict[str, Any] = field(
        default_factory=lambda: {"kernel_size": 3, "groups": 256}
    )


@dataclass
class AudioEncoderConfig:
    """Audio encoder architecture (Section 7.1)."""
    type: str = "ds_cnn"
    width_multiplier: float = 1.0
    temporal_bins: int = 8
    embedding_dim: int = 256


@dataclass
class AlignmentProjectorConfig:
    """Alignment projector architecture (Section 7.1)."""
    projection_dim: int = 128
    input_dim: int = 256


@dataclass
class SyncFeaturesConfig:
    """Synchronization feature extraction (Section 7.1, 7.3)."""
    offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    output_dim: int = 4


@dataclass
class QualityFeaturesConfig:
    """Quality feature dimensions (Section 7.1)."""
    visual_dim: int = 7
    audio_dim: int = 5


@dataclass
class ModalityHeadsConfig:
    """Modality head architecture (Section 7.1)."""
    input_dim: int = 256
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass
class ReliabilityEstimatorConfig:
    """Reliability estimator architecture (Section 7.2)."""
    hidden_dim: int = 128
    dropout: float = 0.1
    gate_temperature: float = 1.0


@dataclass
class FusionClassifierConfig:
    """Fusion classifier architecture (Section 7.1)."""
    hidden_dim: int = 256
    dropout: float = 0.2


@dataclass
class AblationConfig:
    """Ablation flags — [Proposed] decisions toggled for ablation studies."""
    use_sync_features: bool = True
    use_quality_features: bool = True
    use_reliability_gate: bool = True
    fusion_mode: str = "reliability"


@dataclass
class ModelConfig:
    """Top-level model configuration."""
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    alignment_projector: AlignmentProjectorConfig = field(
        default_factory=AlignmentProjectorConfig
    )
    sync_features: SyncFeaturesConfig = field(default_factory=SyncFeaturesConfig)
    quality_features: QualityFeaturesConfig = field(
        default_factory=QualityFeaturesConfig
    )
    modality_heads: ModalityHeadsConfig = field(default_factory=ModalityHeadsConfig)
    reliability_estimator: ReliabilityEstimatorConfig = field(
        default_factory=ReliabilityEstimatorConfig
    )
    fusion_classifier: FusionClassifierConfig = field(
        default_factory=FusionClassifierConfig
    )
    ablation: AblationConfig = field(default_factory=AblationConfig)


@dataclass
class LossConfig:
    """Loss weights (Section 8)."""
    alpha: float = 0.25
    beta: float = 0.10
    gamma: float = 0.05
    eta: float = 0.0
    tau_r: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer parameters."""
    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler parameters."""
    type: str = "cosine_annealing_warm_restarts"
    T_0: int = 10
    T_mult: int = 1
    eta_min: float = 1e-6
    warmup_epochs: int = 3


@dataclass
class AMPConfig:
    """Automatic Mixed Precision settings (Section 10)."""
    enabled: bool = True
    grad_scaler: bool = True


@dataclass
class FreezeConfig:
    """Encoder freeze strategy (Section 8, stage 4)."""
    freeze_encoders_initial_epochs: int = 5
    unfreeze_last_n_blocks: int = 2


@dataclass
class CheckpointConfig:
    """Checkpoint saving parameters."""
    save_dir: str = "artifacts/checkpoints"
    save_every_n_epochs: int = 5
    save_best: bool = True
    metric_for_best: str = "val_auroc"
    save_config_with_checkpoint: bool = True


@dataclass
class EarlyStoppingConfig:
    """Early stopping parameters."""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    stage: str = "unimodal"
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    epochs: dict[str, int] = field(
        default_factory=lambda: {
            "unimodal": 30,
            "fixed_fusion": 30,
            "reliability_fusion": 50,
            "distillation": 20,
        }
    )
    amp: AMPConfig = field(default_factory=AMPConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    seeds: list[int] = field(default_factory=lambda: [42, 123, 7])
    augmentation: dict[str, Any] = field(default_factory=dict)
    class_balance: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""
    baselines: list[dict[str, Any]] = field(default_factory=list)
    degradation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    cross_manipulation: dict[str, Any] = field(default_factory=dict)
    cross_dataset: dict[str, Any] = field(default_factory=dict)
    efficiency: dict[str, Any] = field(default_factory=dict)
    reliability_diagnostics: dict[str, Any] = field(default_factory=dict)


# ── Dataclass construction helpers ───────────────────────────────────────────

def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys.

    For nested dataclass fields the corresponding sub-dict is recursively
    converted.  Non-dataclass fields are passed through as-is.
    """
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_types:
            continue  # skip unknown keys
        ft = field_types[key]
        # Resolve string type annotations
        if isinstance(ft, str):
            ft = eval(ft, globals(), locals())  # noqa: S307 — trusted config schema
        if dataclasses.is_dataclass(ft) and isinstance(value, dict):
            kwargs[key] = _build_dataclass(ft, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# ── Public loader functions ──────────────────────────────────────────────────

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_config(
    config_name: str,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a named config YAML and optionally merge CLI overrides.

    Parameters
    ----------
    config_name:
        Filename (with or without ``.yaml`` extension) under *config_dir*.
    overrides:
        Optional dict of overrides to deep-merge on top.
    config_dir:
        Directory containing configs.  Defaults to ``<project>/configs/``.

    Returns
    -------
    dict[str, Any]
        The merged configuration dictionary.
    """
    config_dir = Path(config_dir) if config_dir else _CONFIGS_DIR
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    raw = load_yaml(config_dir / config_name)
    if overrides:
        raw = merge_configs(raw, overrides)
    return raw


def load_data_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> DataConfig:
    """Load and validate the data pipeline config.

    Returns
    -------
    DataConfig
        Typed data configuration object.
    """
    raw = load_config(
        Path(config_path).stem if config_path else "data_config",
        overrides=overrides,
        config_dir=Path(config_path).parent if config_path else config_dir,
    )
    return _build_dataclass(DataConfig, raw)


def load_model_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> ModelConfig:
    """Load and validate the model architecture config.

    Returns
    -------
    ModelConfig
        Typed model configuration object.
    """
    raw = load_config(
        Path(config_path).stem if config_path else "model_config",
        overrides=overrides,
        config_dir=Path(config_path).parent if config_path else config_dir,
    )
    return _build_dataclass(ModelConfig, raw)


def load_train_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> TrainConfig:
    """Load and validate the training config.

    Returns
    -------
    TrainConfig
        Typed training configuration object.
    """
    raw = load_config(
        Path(config_path).stem if config_path else "train_config",
        overrides=overrides,
        config_dir=Path(config_path).parent if config_path else config_dir,
    )
    return _build_dataclass(TrainConfig, raw)


def load_eval_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> EvalConfig:
    """Load and validate the evaluation config.

    Returns
    -------
    EvalConfig
        Typed evaluation configuration object.
    """
    raw = load_config(
        Path(config_path).stem if config_path else "eval_config",
        overrides=overrides,
        config_dir=Path(config_path).parent if config_path else config_dir,
    )
    return _build_dataclass(EvalConfig, raw)
