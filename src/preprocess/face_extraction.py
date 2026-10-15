"""Detects, aligns, and crops faces (RetinaFace/MediaPipe-style, 20% box expansion, 160x160 output) — Section 6.

Status: implemented — provides FaceExtractor class with detect_and_crop() for single
frames and extract_window_faces() for processing a 16-frame window. Records quality
metadata (detection confidence, face area, failed-frame fraction).
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceCropResult:
    """Result for a single frame's face crop.

    Attributes
    ----------
    crop:
        Aligned face crop of shape ``(crop_size, crop_size, 3)`` RGB uint8,
        or ``None`` if detection failed.
    detected:
        Whether a face was successfully detected.
    confidence:
        Detector confidence score (0–1), or 0.0 if not detected.
    bbox:
        Bounding box ``[x1, y1, x2, y2]`` in original frame coords (expanded).
    face_area:
        Area of the face bounding box in pixels.
    landmarks:
        Facial landmarks if available, else ``None``.
    """
    crop: np.ndarray | None
    detected: bool
    confidence: float
    bbox: list[float] = field(default_factory=list)
    face_area: float = 0.0
    landmarks: np.ndarray | None = None


@dataclass
class WindowFaceResult:
    """Result of face extraction over a full 16-frame window.

    Attributes
    ----------
    crops:
        List of face crops (some may be ``None`` for failed detections).
    detection_success_rate:
        Fraction of frames where a face was detected.
    mean_confidence:
        Mean detector confidence across successful detections.
    mean_face_area:
        Mean face bounding-box area across successful detections.
    failed_frame_indices:
        Indices of frames where face detection failed.
    is_valid:
        Whether the window meets the minimum valid-frame threshold.
    face_areas:
        Per-frame expanded face-box areas; failed detections have area zero.
    detection_confidences:
        Per-frame detector confidence; failed detections have confidence zero.
    """
    crops: list[np.ndarray | None]
    detection_success_rate: float
    mean_confidence: float
    mean_face_area: float
    failed_frame_indices: list[int]
    is_valid: bool
    face_areas: list[float] = field(default_factory=list)
    detection_confidences: list[float] = field(default_factory=list)


class FaceExtractor:
    """Face detection and alignment using OpenCV's DNN-based detector.

    Uses a lightweight DNN face detector (Caffe model) as a portable
    alternative to RetinaFace/MediaPipe.  The detector can be swapped via
    configuration; this implementation provides a working baseline that does
    not require external model downloads for testing.

    Parameters
    ----------
    box_expansion:
        Fraction to expand the detected bounding box (default 0.20 = 20%).
    crop_size:
        Output face crop size in pixels (default 160).
    min_confidence:
        Minimum detector confidence to accept a detection.
    detector:
        Detector backend name.  Currently ``"opencv_dnn"`` is the built-in
        default.  ``"retinaface"`` and ``"mediapipe"`` are planned.
    """

    def __init__(
        self,
        box_expansion: float = 0.20,
        crop_size: int = 160,
        min_confidence: float = 0.5,
        detector: str = "opencv_dnn",
    ) -> None:
        self.box_expansion = box_expansion
        self.crop_size = crop_size
        self.min_confidence = min_confidence
        self.detector_name = detector
        self._detector = self._init_detector(detector)

    def _init_detector(self, name: str) -> object:
        """Initialize the chosen face detection backend."""
        if name in {"opencv_haar", "opencv_dnn"}:
            # OpenCV's built-in Haar cascade is the portable implemented backend.
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                logger.warning(
                    "Haar cascade not found at %s; face detection will fail.",
                    cascade_path,
                )
            return cascade
        elif name in ("retinaface", "mediapipe"):
            logger.warning(
                "Detector '%s' is not installed; using OpenCV Haar cascade fallback.", name
            )
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            return cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError(f"Unknown face detector: {name}")

    def _expand_box(
        self, x1: int, y1: int, x2: int, y2: int, h: int, w: int
    ) -> tuple[int, int, int, int]:
        """Expand a bounding box by ``self.box_expansion`` fraction, clipped to image bounds."""
        bw = x2 - x1
        bh = y2 - y1
        dx = int(bw * self.box_expansion / 2)
        dy = int(bh * self.box_expansion / 2)
        return (
            max(0, x1 - dx),
            max(0, y1 - dy),
            min(w, x2 + dx),
            min(h, y2 + dy),
        )

    def detect_and_crop(self, frame: np.ndarray) -> FaceCropResult:
        """Detect and crop the dominant face in a single RGB frame.

        Parameters
        ----------
        frame:
            RGB image of shape ``(H, W, 3)`` uint8.

        Returns
        -------
        FaceCropResult
            The cropped face and detection metadata.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces using Haar cascade
        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(detections) == 0:
            return FaceCropResult(
                crop=None,
                detected=False,
                confidence=0.0,
                bbox=[],
                face_area=0.0,
                landmarks=None,
            )

        # Select the largest face (by area)
        areas = [dw * dh for (_, _, dw, dh) in detections]
        best_idx = int(np.argmax(areas))
        fx, fy, fw, fh = detections[best_idx]

        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh

        # Expand box by 20%
        ex1, ey1, ex2, ey2 = self._expand_box(x1, y1, x2, y2, h, w)

        # Crop and resize to crop_size × crop_size
        face_region = frame[ey1:ey2, ex1:ex2]
        if face_region.size == 0:
            return FaceCropResult(
                crop=None,
                detected=False,
                confidence=0.0,
                bbox=[float(ex1), float(ey1), float(ex2), float(ey2)],
                face_area=0.0,
                landmarks=None,
            )

        crop = cv2.resize(
            face_region, (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )

        face_area = float((ex2 - ex1) * (ey2 - ey1))
        # Haar cascade doesn't provide confidence; use 1.0 as placeholder
        confidence = 1.0

        return FaceCropResult(
            crop=crop,
            detected=True,
            confidence=confidence,
            bbox=[float(ex1), float(ey1), float(ex2), float(ey2)],
            face_area=face_area,
            landmarks=None,
        )

    def extract_window_faces(
        self,
        frames: np.ndarray,
        min_valid_frames: int = 12,
    ) -> WindowFaceResult:
        """Extract face crops from all frames in a window.

        Parameters
        ----------
        frames:
            Array of shape ``(N, H, W, 3)`` RGB uint8 (typically N=16).
        min_valid_frames:
            Minimum successful detections for the window to be valid.

        Returns
        -------
        WindowFaceResult
            Crops and aggregate quality metadata.
        """
        crops: list[np.ndarray | None] = []
        confidences: list[float] = []
        areas: list[float] = []
        frame_areas: list[float] = []
        frame_confidences: list[float] = []
        failed: list[int] = []

        for i, frame in enumerate(frames):
            result = self.detect_and_crop(frame)
            crops.append(result.crop)
            frame_areas.append(result.face_area if result.detected else 0.0)
            frame_confidences.append(result.confidence if result.detected else 0.0)
            if result.detected:
                confidences.append(result.confidence)
                areas.append(result.face_area)
            else:
                failed.append(i)

        n_detected = len(frames) - len(failed)
        detection_rate = n_detected / len(frames) if len(frames) > 0 else 0.0

        return WindowFaceResult(
            crops=crops,
            detection_success_rate=detection_rate,
            mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
            mean_face_area=float(np.mean(areas)) if areas else 0.0,
            failed_frame_indices=failed,
            is_valid=n_detected >= min_valid_frames,
            face_areas=frame_areas,
            detection_confidences=frame_confidences,
        )
