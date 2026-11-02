"""Run one uploaded clip through the configured detector without retaining media."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.manifest_dataset import ManifestAVDataset, dataset_roots_from_config
from src.evaluate.metrics import find_optimal_threshold
from src.models.full_model import FullModel
from src.utils.config_loader import (
    load_data_config,
    load_eval_config,
    load_model_config,
    load_train_config,
    load_yaml,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InferenceInputError(ValueError):
    """An uploaded clip cannot be used under the configured demo input contract."""


def _sha256_file(path: Path) -> str:
    """Hash a local file in bounded memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for a small configuration or manifest file."""
    return hashlib.sha256(value).hexdigest()


def _project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root rather than the shell cwd."""
    resolved = Path(path).expanduser()
    return resolved.resolve() if resolved.is_absolute() else (PROJECT_ROOT / resolved).resolve()


class ClipPredictionService:
    """Load one full AV checkpoint, select its operating point on validation, and score uploads."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        data_config_path: str | Path = "configs/data_config.yaml",
        model_config_path: str | Path = "configs/model_config.yaml",
        train_config_path: str | Path = "configs/train_config.yaml",
        eval_config_path: str | Path = "configs/eval_config.yaml",
        manifest_dir: str | Path = "manifests/splits",
        web_config_path: str | Path = "configs/web_app_config.yaml",
    ) -> None:
        """Initialize config and paths; model and validation threshold load lazily."""
        self.checkpoint_path = _project_path(checkpoint_path)
        self.data_config_path = _project_path(data_config_path)
        self.model_config_path = _project_path(model_config_path)
        self.train_config_path = _project_path(train_config_path)
        self.eval_config_path = _project_path(eval_config_path)
        self.manifest_dir = _project_path(manifest_dir)
        self.web_config_path = _project_path(web_config_path)

        self.data_config = load_data_config(config_path=self.data_config_path)
        self.model_config = load_model_config(config_path=self.model_config_path)
        self.train_config = load_train_config(config_path=self.train_config_path)
        self.eval_config = load_eval_config(config_path=self.eval_config_path)
        self.web_config = load_yaml(self.web_config_path).get("server", {})
        self.max_upload_bytes = int(self.web_config["max_upload_mb"]) * 1024 * 1024
        self.max_clip_duration_s = float(self.web_config["max_clip_duration_s"])
        self.max_frame_pixels = int(self.web_config["max_frame_pixels"])
        self.allowed_extensions = {
            str(suffix).lower()
            for suffix in self.web_config["allowed_extensions"]
        }
        self.threshold_cache_dir = _project_path(self.web_config["threshold_cache_dir"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: FullModel | None = None
        self.checkpoint_sha256: str | None = None
        self.threshold: float | None = None
        self._lock = threading.RLock()

    @property
    def window_duration_s(self) -> float:
        """Return the configured inference window duration."""
        return float(self.data_config.decode.window_duration_s)

    def load_model(self) -> None:
        """Load a full audio-visual checkpoint and reject audio-only baselines."""
        with self._lock:
            if self.model is not None:
                return
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            if not any(key.startswith("audio_encoder.") for key in state):
                raise ValueError(
                    "The browser demo needs a full audio-visual checkpoint. "
                    "Choose artifacts/checkpoints/reliability_fusion/best_model.pt."
                )

            # All visual encoder weights are in the checkpoint; avoid downloading
            # ImageNet weights just to overwrite them during state restoration.
            self.model_config.visual_encoder.pretrained = False
            model = FullModel(self.model_config)
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
            self.model = model
            self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
            logger.info("Loaded local AV checkpoint %s on %s", self.checkpoint_path.name, self.device)

    def status(self) -> dict[str, Any]:
        """Describe local readiness without loading or evaluating any dataset rows."""
        validation_manifest = self.manifest_dir / "val_manifest.csv"
        threshold_cached = False
        if self.threshold is not None:
            threshold_cached = True
        elif self.checkpoint_sha256 is not None and validation_manifest.is_file():
            threshold_cached = self._threshold_cache_path(validation_manifest).is_file()
        return {
            "ready": self.model is not None,
            "checkpoint": self.checkpoint_path.name,
            "threshold_ready": threshold_cached,
            "threshold_source": "FakeAVCeleb validation",
            "device": str(self.device),
            "window_duration_s": self.window_duration_s,
            "max_upload_mb": self.max_upload_bytes // (1024 * 1024),
            "max_clip_duration_s": self.max_clip_duration_s,
        }

    def _threshold_cache_path(self, validation_manifest: Path) -> Path:
        """Return the ignored local cache path for this checkpoint and validation manifest."""
        assert self.checkpoint_sha256 is not None
        key = _sha256_bytes(
            "|".join(
                (
                    self.checkpoint_sha256,
                    _sha256_file(validation_manifest),
                    _sha256_file(self.data_config_path),
                    _sha256_file(self.model_config_path),
                    _sha256_file(self.eval_config_path),
                    str(self.eval_config.threshold.get("method", "youden_j")),
                )
            ).encode("utf-8")
        )[:20]
        self.threshold_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.threshold_cache_dir / f"validation_threshold_{key}.json"

    def _fit_validation_threshold(self) -> float:
        """Choose the app's decision threshold using only the configured validation manifest."""
        if self.model is None:
            raise RuntimeError("Load the model before fitting its validation threshold")
        validation_manifest = self.manifest_dir / "val_manifest.csv"
        if not validation_manifest.is_file():
            raise FileNotFoundError(
                f"Validation manifest not found: {validation_manifest}. Build the data manifests first."
            )
        cache_path = self._threshold_cache_path(validation_manifest)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.threshold = float(cached["threshold"])
            return self.threshold

        roots = dataset_roots_from_config(self.data_config)
        if "fakeavceleb" not in roots:
            raise RuntimeError(
                "The app needs the local FakeAVCeleb validation data to choose its operating threshold. "
                "Set FAKEAVCELEB_ROOT or datasets.fakeavceleb.root."
            )
        validation_data = ManifestAVDataset(
            validation_manifest,
            roots,
            _project_path(self.data_config.cache.cache_dir) / "features",
            self.data_config,
            allowed_datasets={"fakeavceleb"},
        )
        batch_size = max(1, int(self.train_config.batch_size))
        loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
        logits: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        self.model.eval()
        with torch.inference_mode():
            for batch in loader:
                output = self.model(
                    frames=batch["frames"].to(self.device, dtype=torch.float32),
                    mel_spec=batch["mel_spec"].to(self.device, dtype=torch.float32),
                    quality_v=batch["quality_v"].to(self.device, dtype=torch.float32),
                    quality_a=batch["quality_a"].to(self.device, dtype=torch.float32),
                    visual_missing=batch["visual_missing"].to(self.device, dtype=torch.float32),
                    audio_missing=batch["audio_missing"].to(self.device, dtype=torch.float32),
                )
                logits.append(output["logit_f"].float().cpu().reshape(-1))
                targets.append(batch["target"].float().cpu().reshape(-1))
        if not targets:
            raise ValueError("FakeAVCeleb validation manifest produced no usable clips")
        y = torch.cat(targets).numpy()
        probabilities = torch.sigmoid(torch.cat(logits)).numpy()
        method = str(self.eval_config.threshold.get("method", "youden_j"))
        self.threshold = find_optimal_threshold(y, probabilities, method=method)
        cache_path.write_text(
            json.dumps(
                {
                    "threshold": self.threshold,
                    "method": method,
                    "source": "FakeAVCeleb validation",
                    "checkpoint_sha256": self.checkpoint_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Fit demo threshold %.6f on FakeAVCeleb validation", self.threshold)
        return self.threshold

    def _inspect_video(self, path: Path) -> dict[str, float | int]:
        """Check basic container metadata against the configured short-clip limit."""
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise InferenceInputError("Could not open this video. Try an H.264 MP4 clip.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
            raise InferenceInputError("The video has invalid frame-rate or size metadata.")
        duration = frames / fps
        if duration < self.window_duration_s:
            raise InferenceInputError(
                f"The clip must be at least {self.window_duration_s:g} seconds long."
            )
        if duration > self.max_clip_duration_s:
            raise InferenceInputError(
                f"The clip is {duration:.1f} seconds long; the local demo limit is "
                f"{self.max_clip_duration_s:g} seconds. Trim it and try again."
            )
        if width * height > self.max_frame_pixels:
            raise InferenceInputError(
                f"The video resolution is {width}×{height}; the local demo accepts up to "
                f"{self.max_frame_pixels:,} pixels per frame."
            )
        return {"duration_s": duration, "width": width, "height": height}

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        """Preprocess one local video, run the detector, and return display-ready scores."""
        path = Path(video_path).resolve()
        if path.suffix.lower() not in self.allowed_extensions:
            raise InferenceInputError(
                "Unsupported video format. Use MP4, MOV, MKV, AVI, or WebM."
            )
        if not path.is_file():
            raise InferenceInputError("The uploaded video could not be found.")
        if path.stat().st_size > self.max_upload_bytes:
            raise InferenceInputError(
                f"The file exceeds the {self.max_upload_bytes // (1024 * 1024)} MB upload limit."
            )
        video_info = self._inspect_video(path)

        with self._lock:
            self.load_model()
            threshold = self.threshold if self.threshold is not None else self._fit_validation_threshold()
            assert self.model is not None
            with TemporaryDirectory(prefix="av-detector-demo-") as temp_dir:
                temp_root = Path(temp_dir)
                manifest_path = temp_root / "upload_manifest.csv"
                pd.DataFrame(
                    [
                        {
                            "clip_id": "uploaded_clip",
                            "dataset": "upload",
                            "source_identity": "local_upload",
                            "file_path": path.name,
                            "clip_fake": 0,
                        }
                    ]
                ).to_csv(manifest_path, index=False)
                dataset = ManifestAVDataset(
                    manifest_path,
                    {"upload": path.parent},
                    temp_root / "features",
                    self.data_config,
                    allowed_datasets={"upload"},
                )
                batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
                visual_missing = bool(batch["visual_missing"].reshape(-1)[0].item() >= 0.5)
                audio_missing = bool(batch["audio_missing"].reshape(-1)[0].item() >= 0.5)
                if visual_missing and audio_missing:
                    raise InferenceInputError(
                        f"No usable face or speech was found in the first {self.window_duration_s:g} seconds. "
                        "Try a clearer clip with a visible face and audible speech."
                    )
                tensors = {
                    key: value.to(self.device, dtype=torch.float32)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                with torch.inference_mode():
                    output = self.model(
                        frames=tensors["frames"],
                        mel_spec=tensors["mel_spec"],
                        quality_v=tensors["quality_v"],
                        quality_a=tensors["quality_a"],
                        visual_missing=tensors["visual_missing"],
                        audio_missing=tensors["audio_missing"],
                    )

                p_fake = float(output["p_f"].reshape(-1)[0].cpu().item())
                p_visual = float(output["p_v"].reshape(-1)[0].cpu().item())
                p_audio = float(output["p_a"].reshape(-1)[0].cpu().item())
                w_visual = float(output["w_v"].reshape(-1)[0].cpu().item())
                w_audio = float(output["w_a"].reshape(-1)[0].cpu().item())
                quality_v = batch["quality_v"].reshape(-1, 7)[0].tolist()
                quality_a = batch["quality_a"].reshape(-1, 5)[0].tolist()

        return {
            "prediction": "fake" if p_fake >= threshold else "real",
            "p_fake": p_fake,
            "threshold": float(threshold),
            "threshold_source": "FakeAVCeleb validation",
            "threshold_method": str(self.eval_config.threshold.get("method", "youden_j")),
            "scores": {"fused": p_fake, "visual": p_visual, "audio": p_audio},
            "reliability": {"visual": w_visual, "audio": w_audio},
            "quality": {
                "face_detection_rate": float(quality_v[0]),
                "voice_activity_fraction": float(quality_a[0]),
                "visual_missing": visual_missing,
                "audio_missing": audio_missing,
            },
            "window_duration_s": self.window_duration_s,
            "video": video_info,
            "checkpoint": self.checkpoint_path.name,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
