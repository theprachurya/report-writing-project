"""Deterministic fixed-window feature extraction and on-disk cache creation."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.decode.audio_decoder import decode_audio_window
from src.decode.video_decoder import decode_video_window
from src.preprocess.audio_features import compute_log_mel_spectrogram
from src.preprocess.face_extraction import FaceExtractor
from src.preprocess.quality_features import compute_audio_quality, compute_visual_quality
from src.preprocess.alignment import TemporalAligner

logger = logging.getLogger(__name__)


def preprocess_manifest_row(
    row: Any,
    dataset_root: str | Path,
    output_path: str | Path,
    data_config: Any,
) -> Path:
    """Decode one manifest clip, extract contract features, and atomically cache them."""
    root = Path(dataset_root).resolve()
    relative_path = Path(str(row["file_path"]))
    clip_path = (root / relative_path).resolve()
    if not clip_path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes its configured dataset root: {relative_path}")
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip from manifest does not exist: {clip_path}")

    cfg = data_config
    video = None
    face_valid_mask = np.zeros(cfg.decode.frames_per_window, dtype=bool)
    face_areas = np.zeros(cfg.decode.frames_per_window, dtype=np.float32)
    detection_confidences = np.zeros(cfg.decode.frames_per_window, dtype=np.float32)
    try:
        video = decode_video_window(
            clip_path,
            window_duration_s=cfg.decode.window_duration_s,
            frames_per_window=cfg.decode.frames_per_window,
            target_fps=cfg.decode.target_fps,
        )
        face_extractor = FaceExtractor(
            box_expansion=cfg.face.box_expansion,
            crop_size=cfg.face.crop_size,
            detector=cfg.face.detector,
        )
        faces = face_extractor.extract_window_faces(
            video.frames, min_valid_frames=cfg.face.min_valid_frames
        )
        face_valid_mask = np.asarray([crop is not None for crop in faces.crops], dtype=bool)
        face_areas = np.asarray(faces.face_areas, dtype=np.float32)
        detection_confidences = np.asarray(faces.detection_confidences, dtype=np.float32)
        black_crop = np.zeros((cfg.face.crop_size, cfg.face.crop_size, 3), dtype=np.uint8)
        crops = np.stack([crop if crop is not None else black_crop for crop in faces.crops])
        quality_v = compute_visual_quality(
            faces.crops,
            face_areas=faces.face_areas,
            detection_confidences=faces.detection_confidences,
        ).to_vector()
        visual_missing = float(not faces.is_valid)
    except Exception as exc:
        logger.warning("Visual extraction failed for %s: %s", clip_path, exc)
        crops = np.zeros(
            (cfg.decode.frames_per_window, cfg.face.crop_size, cfg.face.crop_size, 3),
            dtype=np.uint8,
        )
        quality_v = np.zeros(cfg.quality_features.visual_dim, dtype=np.float32)
        visual_missing = 1.0

    audio_missing = 0.0
    waveform = np.zeros(
        int(cfg.audio.sample_rate * cfg.decode.window_duration_s), dtype=np.float32
    )
    try:
        audio = decode_audio_window(
            clip_path,
            window_duration_s=cfg.decode.window_duration_s,
            target_sr=cfg.audio.sample_rate,
            peak_normalize=False,
        )
        quality_a = compute_audio_quality(
            audio.waveform,
            sample_rate=cfg.audio.sample_rate,
            expected_duration_s=cfg.decode.window_duration_s,
        ).to_vector()
        quality_a[2] = audio.clipped_sample_ratio
        waveform = audio.waveform
        if cfg.audio.peak_normalize and audio.peak_amplitude > 0:
            waveform = waveform / audio.peak_amplitude
        mel = compute_log_mel_spectrogram(
            waveform,
            sample_rate=cfg.audio.sample_rate,
            n_mels=cfg.mel.n_mels,
            window_length_ms=cfg.mel.window_length_ms,
            hop_length_ms=cfg.mel.hop_length_ms,
            f_min=cfg.mel.f_min,
            f_max=cfg.mel.f_max,
        )
        audio_missing = float(quality_a[0] < cfg.validity.min_vad_fraction)
        mel_spec = mel.spectrogram
    except Exception as exc:
        logger.warning("Audio extraction failed for %s: %s", clip_path, exc)
        expected_samples = int(cfg.audio.sample_rate * cfg.decode.window_duration_s)
        mel = compute_log_mel_spectrogram(
            np.zeros(expected_samples, dtype=np.float32),
            sample_rate=cfg.audio.sample_rate,
            n_mels=cfg.mel.n_mels,
            window_length_ms=cfg.mel.window_length_ms,
            hop_length_ms=cfg.mel.hop_length_ms,
            f_min=cfg.mel.f_min,
            f_max=cfg.mel.f_max,
        )
        mel_spec = mel.spectrogram
        quality_a = np.zeros(cfg.quality_features.audio_dim, dtype=np.float32)
        audio_missing = 1.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=output_path.parent, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        np.savez_compressed(
            temp_file,
            frames=crops.astype(np.uint8),
            face_valid_mask=face_valid_mask,
            face_areas=face_areas,
            detection_confidences=detection_confidences,
            audio_waveform=np.asarray(waveform, dtype=np.float32),
            mel_spec=np.asarray(mel_spec, dtype=np.float32),
            quality_v=np.asarray(quality_v, dtype=np.float32),
            quality_a=np.asarray(quality_a, dtype=np.float32),
            visual_missing=np.float32(visual_missing),
            audio_missing=np.float32(audio_missing),
            frame_timestamps_s=(video.timestamps_s if video is not None else np.zeros(cfg.decode.frames_per_window, dtype=np.float64)),
            timestamp_transform=np.asarray(
                TemporalAligner(
                    n_bins=cfg.alignment.temporal_bins,
                    window_duration_s=cfg.decode.window_duration_s,
                ).compute_timestamp_transform(),
                dtype=np.float64,
            ),
            dropped_frame_indices=(np.asarray(video.dropped_frame_indices, dtype=np.int64) if video is not None else np.arange(cfg.decode.frames_per_window, dtype=np.int64)),
        )
    os.replace(temp_path, output_path)
    return output_path
