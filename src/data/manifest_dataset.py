"""Load cached or locally extracted AV features from a leakage-safe CSV manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def dataset_roots_from_config(data_config: Any) -> dict[str, Path]:
    """Return configured dataset roots that resolve to existing local folders."""
    roots: dict[str, Path] = {}
    datasets = getattr(data_config, "datasets", {})
    for name, spec in datasets.items():
        raw_root = spec.get("root") if isinstance(spec, dict) else getattr(spec, "root", None)
        if not raw_root or (isinstance(raw_root, str) and "${" in raw_root):
            continue
        root = Path(raw_root).expanduser()
        if root.is_dir():
            roots[str(name).lower()] = root.resolve()
    return roots


class ManifestAVDataset(Dataset[dict[str, torch.Tensor]]):
    """Read a split manifest and return fixed-shape AV tensors for training.

    Features are loaded from ``cache_dir`` when available. A cache miss invokes
    deterministic local preprocessing and stores the result outside the repo's
    tracked manifests. The input CSV must already be a committed split manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_roots: dict[str, str | Path],
        cache_dir: str | Path,
        data_config: Any,
        *,
        training: bool = False,
        modality_dropout_prob: float = 0.0,
        augmentation_config: dict[str, Any] | None = None,
        allowed_datasets: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Split manifest not found: {self.manifest_path}. Run `python scripts/run_data_audit.py` "
                "with a licensed local dataset root before training."
            )
        self.rows = pd.read_csv(self.manifest_path)
        required = {"clip_id", "dataset", "file_path", "source_identity", "clip_fake"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}")
        if allowed_datasets is not None:
            allowed = {name.lower() for name in allowed_datasets}
            self.rows = self.rows[self.rows["dataset"].astype(str).str.lower().isin(allowed)].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No permitted dataset rows remain in split manifest: {self.manifest_path}")

        self.dataset_roots = {k.lower(): Path(v).resolve() for k, v in dataset_roots.items()}
        self.cache_dir = Path(cache_dir)
        self.data_config = data_config
        self.training = training
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.augmentation_config = augmentation_config or {}

    def __len__(self) -> int:
        """Return the number of clips in this manifest."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one cached feature window and its label/missingness metadata."""
        row = self.rows.iloc[index]
        dataset_name = str(row["dataset"]).lower()
        clip_id = str(row["clip_id"])
        cache_key = f"{dataset_name}:{clip_id}"
        cache_name = hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".npz"
        cache_path = self.cache_dir / cache_name
        # Fallback to legacy cache key if existing cache was generated with clip_id only
        if not cache_path.is_file():
            legacy_name = hashlib.sha256(clip_id.encode("utf-8")).hexdigest() + ".npz"
            legacy_path = self.cache_dir / legacy_name
            if legacy_path.is_file():
                cache_path = legacy_path

        aug_cfg = self.augmentation_config
        video_cfg = aug_cfg.get("video", {}) if isinstance(aug_cfg, dict) else {}
        audio_cfg = aug_cfg.get("audio", {}) if isinstance(aug_cfg, dict) else {}
        needed_training_cache_fields = set()
        if self.training and video_cfg.get("enabled", False):
            needed_training_cache_fields.update({"face_valid_mask", "face_areas", "detection_confidences"})
        if self.training and audio_cfg.get("enabled", False):
            needed_training_cache_fields.add("audio_waveform")
        cache_needs_refresh = False
        if cache_path.is_file() and needed_training_cache_fields:
            with np.load(cache_path, allow_pickle=False) as cached:
                missing_cache_fields = needed_training_cache_fields - set(cached.files)
            cache_needs_refresh = bool(missing_cache_fields)
        if not cache_path.is_file() or cache_needs_refresh:
            from src.preprocess.feature_cache import preprocess_manifest_row

            if dataset_name not in self.dataset_roots:
                raise ValueError(
                    f"No existing local root configured for dataset {dataset_name!r}; "
                    "set its *_ROOT environment variable or edit configs/data_config.yaml."
                )
            preprocess_manifest_row(
                row=row,
                dataset_root=self.dataset_roots[dataset_name],
                output_path=cache_path,
                data_config=self.data_config,
            )

        with np.load(cache_path, allow_pickle=False) as cached:
            frames = np.asarray(cached["frames"], dtype=np.float32)
            mel_spec = np.asarray(cached["mel_spec"], dtype=np.float32)
            quality_v = np.asarray(cached["quality_v"], dtype=np.float32)
            quality_a = np.asarray(cached["quality_a"], dtype=np.float32)
            waveform = (
                np.asarray(cached["audio_waveform"], dtype=np.float32)
                if "audio_waveform" in cached.files else None
            )
            face_valid_mask = (
                np.asarray(cached["face_valid_mask"], dtype=bool)
                if "face_valid_mask" in cached.files else np.any(frames != 0, axis=(1, 2, 3))
            )
            face_areas = (
                np.asarray(cached["face_areas"], dtype=np.float32)
                if "face_areas" in cached.files else np.zeros(len(frames), dtype=np.float32)
            )
            detection_confidences = (
                np.asarray(cached["detection_confidences"], dtype=np.float32)
                if "detection_confidences" in cached.files else np.zeros(len(frames), dtype=np.float32)
            )
            visual_missing = float(cached["visual_missing"])
            audio_missing = float(cached["audio_missing"])

        # Training degradation is modality-specific and recomputes the affected
        # quality vector so the reliability gate sees current input quality.
        enabled_modalities = []
        if self.training and video_cfg.get("enabled", False) and visual_missing < 0.5:
            enabled_modalities.append("video")
        if self.training and audio_cfg.get("enabled", False) and audio_missing < 0.5 and waveform is not None:
            enabled_modalities.append("audio")
        if enabled_modalities:
            from src.preprocess.augmentation import AudioAugmentor, VideoAugmentor
            from src.preprocess.audio_features import compute_log_mel_spectrogram
            from src.preprocess.quality_features import compute_audio_quality, compute_visual_quality

            # Incorporate DataLoader worker ID into RNG seed for decorrelated streams
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            base_seed = int(torch.randint(0, 2**31 - 1, ()).item())
            rng_seed = (base_seed + worker_id * 10007) % (2**31 - 1)
            rng = np.random.default_rng(rng_seed)

            selected_modality = enabled_modalities[int(rng.integers(0, len(enabled_modalities)))]
            if selected_modality == "video":
                frames, _ = VideoAugmentor(
                    resize_prob=float(video_cfg.get("resize_prob", 0.0)),
                    blur_prob=float(video_cfg.get("blur_prob", 0.0)),
                    frame_drop_prob=float(video_cfg.get("frame_drop_prob", 0.0)),
                    reencode_prob=float(video_cfg.get("reencode_prob", 0.0)),
                    noise_prob=float(video_cfg.get("noise_prob", 0.0)),
                    brightness_prob=float(video_cfg.get("brightness_prob", 0.0)),
                    rng=rng,
                )(frames.astype(np.uint8))
                # Re-derive face validity mask after video augmentations (e.g. frame drop / zeroing)
                face_valid_mask = face_valid_mask & np.any(frames != 0, axis=(1, 2, 3))
                min_valid = getattr(self.data_config.face, "min_valid_frames", 12)
                visual_missing = float(np.sum(face_valid_mask) < min_valid)
                crop_list = [
                    frame if is_valid else None
                    for frame, is_valid in zip(frames, face_valid_mask)
                ]
                quality_v = compute_visual_quality(
                    crop_list,
                    face_areas=face_areas.tolist(),
                    detection_confidences=detection_confidences.tolist(),
                ).to_vector()
            elif waveform is not None:
                waveform, _ = AudioAugmentor(
                    noise_prob=float(audio_cfg.get("noise_prob", 0.0)),
                    lowpass_prob=float(audio_cfg.get("lowpass_prob", 0.0)),
                    dropout_prob=float(audio_cfg.get("dropout_prob", 0.0)),
                    reverb_prob=float(audio_cfg.get("reverb_prob", 0.0)),
                    rng=rng,
                )(waveform, sample_rate=self.data_config.audio.sample_rate)
                audio_quality = compute_audio_quality(
                    waveform,
                    sample_rate=self.data_config.audio.sample_rate,
                    expected_duration_s=self.data_config.decode.window_duration_s,
                )
                quality_a = audio_quality.to_vector()
                mel_spec = compute_log_mel_spectrogram(
                    waveform,
                    sample_rate=self.data_config.audio.sample_rate,
                    n_mels=self.data_config.mel.n_mels,
                    window_length_ms=self.data_config.mel.window_length_ms,
                    hop_length_ms=self.data_config.mel.hop_length_ms,
                    f_min=self.data_config.mel.f_min,
                    f_max=self.data_config.mel.f_max,
                ).spectrogram
                audio_missing = float(
                    audio_quality.vad_fraction < self.data_config.validity.min_vad_fraction
                )

        if self.training and self.modality_dropout_prob > 0:
            draw = float(torch.rand(()))
            if draw < self.modality_dropout_prob:
                if float(torch.rand(())) < 0.5:
                    visual_missing = 1.0
                else:
                    audio_missing = 1.0

        normalization = self.data_config.visual_normalization
        mean = np.asarray(normalization.mean, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.asarray(normalization.std, dtype=np.float32).reshape(1, 1, 1, 3)
        frames = ((frames.astype(np.float32) / 255.0) - mean) / std
        frames = frames.astype(np.float32)
        if visual_missing >= 0.5:
            frames = np.zeros_like(frames)
        if audio_missing >= 0.5:
            mel_spec = np.zeros_like(mel_spec)

        return {
            "frames": torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).permute(0, 3, 1, 2),
            "mel_spec": torch.from_numpy(np.ascontiguousarray(mel_spec, dtype=np.float32)).unsqueeze(0),
            "quality_v": torch.from_numpy(np.asarray(quality_v, dtype=np.float32)),
            "quality_a": torch.from_numpy(np.asarray(quality_a, dtype=np.float32)),
            "visual_missing": torch.tensor([visual_missing], dtype=torch.float32),
            "audio_missing": torch.tensor([audio_missing], dtype=torch.float32),
            "target": torch.tensor([float(row["clip_fake"])], dtype=torch.float32),
        }
