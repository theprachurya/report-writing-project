"""Audio-only MLAAD-tiny manifest and inference helpers for out-of-domain evaluation."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.decode.audio_decoder import decode_audio_window
from src.preprocess.audio_features import compute_log_mel_spectrogram

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioEvalRecord:
    """One audio item with its dataset label and source-utterance group."""

    clip_id: str
    file_path: Path
    target: int
    source_group: str
    generator: str


def _safe_relative_path(value: str) -> Path:
    """Convert a dataset metadata path to a safe path relative to the local root."""
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Dataset metadata contains an unsafe relative path: {value!r}")
    return Path(*path.parts)


def build_mlaad_audio_records(
    dataset_root: str | Path,
    language: str = "de",
) -> list[AudioEvalRecord]:
    """Build bona-fide/spoof records and retain source groups from MLAAD metadata.

    MLAAD-tiny omits the source audio for some spoof rows. Keep those spoof
    records, include each locally available bona-fide file once, and retain
    source utterance IDs so confidence intervals can account for multiple TTS
    outputs derived from the same source. A real recording is linked to its
    source group only when its basename maps unambiguously.
    """
    root = Path(dataset_root).expanduser().resolve()
    language = language.lower()
    fake_root = root / "fake" / language
    original_root = root / "original" / language
    if not fake_root.is_dir() or not original_root.is_dir():
        raise FileNotFoundError(
            f"Expected MLAAD-tiny folders fake/{language} and original/{language} under {root}. "
            "Run scripts/download_mlaad_tiny.py first."
        )

    original_by_name: dict[str, list[Path]] = {}
    for path in sorted(original_root.rglob("*.wav")):
        original_by_name.setdefault(path.name, []).append(path)

    fake_records: list[AudioEvalRecord] = []
    original_groups_by_path: dict[Path, set[str]] = {}
    seen_fake_paths: set[Path] = set()
    missing_original_rows = 0
    metadata_files = sorted(fake_root.glob("*/meta.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No per-generator meta.csv files found under {fake_root}")

    for metadata_path in metadata_files:
        with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            required = {"path", "original_file", "language", "model_name"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Unexpected MLAAD-tiny metadata columns in {metadata_path}: {reader.fieldnames}"
                )
            for row in reader:
                if str(row.get("language", "")).strip().lower() != language:
                    continue
                fake_relative = _safe_relative_path(str(row["path"]))
                fake_path = root / fake_relative
                original_reference = _safe_relative_path(str(row["original_file"]))
                original_matches = original_by_name.get(original_reference.name, [])
                if len(original_matches) > 1:
                    raise FileNotFoundError(
                        f"Expected at most one local original for {row['original_file']!r}; found {len(original_matches)}"
                    )
                if not fake_path.is_file():
                    raise FileNotFoundError(f"MLAAD-tiny spoof file is missing: {fake_path}")
                if fake_path in seen_fake_paths:
                    raise ValueError(f"Duplicate spoof path in MLAAD-tiny metadata: {fake_path}")
                seen_fake_paths.add(fake_path)

                source_group = original_reference.as_posix()
                if original_matches:
                    original_groups_by_path.setdefault(original_matches[0], set()).add(source_group)
                else:
                    missing_original_rows += 1
                generator = str(row.get("model_name", "")).strip() or metadata_path.parent.name
                key = fake_relative.as_posix()
                fake_records.append(
                    AudioEvalRecord(
                        clip_id=f"mlaad_fake_{key}",
                        file_path=fake_path,
                        target=1,
                        source_group=source_group,
                        generator=generator,
                    )
                )

    if not fake_records:
        raise ValueError(f"No MLAAD-tiny samples matched language {language!r}")
    records = list(fake_records)
    for original_path in sorted(path for paths in original_by_name.values() for path in paths):
        source_groups = original_groups_by_path.get(original_path, set())
        if len(source_groups) > 1:
            raise ValueError(
                f"Local original {original_path} maps to multiple source IDs: {sorted(source_groups)}"
            )
        source_group = next(iter(source_groups)) if source_groups else f"local:{original_path.relative_to(root).as_posix()}"
        relative_path = original_path.relative_to(root).as_posix()
        records.append(
            AudioEvalRecord(
                clip_id=f"mlaad_real_{relative_path}",
                file_path=original_path,
                target=0,
                source_group=source_group,
                generator="bona_fide",
            )
        )
    if {record.target for record in records} != {0, 1}:
        raise ValueError("MLAAD-tiny external evaluation requires bona-fide and spoof classes")
    logger.info(
        "Built %d audio records (%d spoof, %d bona-fide; %d spoof rows lack source audio) for language %s",
        len(records),
        len(fake_records),
        len(records) - len(fake_records),
        missing_original_rows,
        language,
    )
    return records


class MlaadAudioDataset(Dataset[dict[str, Any]]):
    """Decode consistent 2-second log-Mel windows from MLAAD-tiny audio records."""

    def __init__(
        self,
        records: list[AudioEvalRecord],
        data_config: Any,
        start_seconds: float = 0.0,
    ) -> None:
        self.records = records
        self.data_config = data_config
        self.start_seconds = float(start_seconds)

    def __len__(self) -> int:
        """Return the number of audio windows."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode and featurize one audio window using the project's configured contract."""
        record = self.records[index]
        cfg = self.data_config
        decoded = decode_audio_window(
            record.file_path,
            start_s=self.start_seconds,
            window_duration_s=cfg.decode.window_duration_s,
            target_sr=cfg.audio.sample_rate,
            peak_normalize=cfg.audio.peak_normalize,
        )
        mel = compute_log_mel_spectrogram(
            decoded.waveform,
            sample_rate=cfg.audio.sample_rate,
            n_mels=cfg.mel.n_mels,
            window_length_ms=cfg.mel.window_length_ms,
            hop_length_ms=cfg.mel.hop_length_ms,
            f_min=cfg.mel.f_min,
            f_max=cfg.mel.f_max,
        ).spectrogram
        return {
            "mel_spec": torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).unsqueeze(0),
            "target": torch.tensor(record.target, dtype=torch.float32),
            "source_group": record.source_group,
            "generator": record.generator,
            "clip_id": record.clip_id,
        }


def predict_audio_head(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return audio-head logits and labels without running the visual or fusion branches."""
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    groups: list[str] = []
    generators: list[str] = []
    model.eval()
    if hasattr(model, "audio_encoder") and hasattr(model, "modality_heads"):
        audio_encoder = model.audio_encoder
        audio_head = model.modality_heads.audio_head
    elif hasattr(model, "encoder") and hasattr(model, "head"):
        # The staged unimodal baseline stores the same branch as encoder/head.
        audio_encoder = model.encoder
        audio_head = model.head
    else:
        raise TypeError("Expected a full model or an audio-only encoder/head model")
    with torch.inference_mode():
        for batch in loader:
            mel_spec = batch["mel_spec"].to(device=device, dtype=torch.float32)
            z_a = audio_encoder(mel_spec)
            audio_output = audio_head(z_a)
            logits.append(audio_output["logit"].float().cpu().reshape(-1))
            targets.append(batch["target"].float().cpu().reshape(-1))
            batch_size = int(batch["target"].shape[0])
            groups.extend(
                str(value) for value in batch.get("source_group", ["unknown"] * batch_size)
            )
            generators.extend(
                str(value) for value in batch.get("generator", ["unknown"] * batch_size)
            )
    if not targets:
        raise ValueError("Audio evaluation loader produced no samples")
    return {
        "logits": torch.cat(logits).numpy(),
        "targets": torch.cat(targets).numpy(),
        "source_groups": np.asarray(groups, dtype=str),
        "generators": np.asarray(generators, dtype=str),
    }


def bootstrap_grouped_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Estimate an AUROC interval by resampling original-source groups with replacement."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(targets).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    if not (len(y) == len(p) == len(groups)):
        raise ValueError("targets, probabilities, and group_ids must have the same length")
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Grouped AUROC bootstrap requires non-empty data with both classes")

    by_group: dict[str, np.ndarray] = {}
    for group in np.unique(groups):
        by_group[str(group)] = np.flatnonzero(groups == group)
    group_names = np.asarray(sorted(by_group), dtype=str)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(max(1, int(n_iterations))):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in sampled_groups])
        if len(np.unique(y[indices])) == 2:
            scores.append(float(roc_auc_score(y[indices], p[indices])))
    if not scores:
        return {"lower": 0.0, "upper": 0.0, "valid_replicates": 0}
    tail = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(scores, [tail, 1.0 - tail])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": len(scores),
    }
