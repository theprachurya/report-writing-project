"""Feature extraction and augmentation subpackage.

Provides face extraction, audio feature computation (log-Mel spectrograms),
modality quality proxies, temporal alignment, and training augmentation.
See AGENTS.md for build order.
"""

from __future__ import annotations

# Core preprocessing modules
from src.preprocess.alignment import AlignmentResult, TemporalAligner
from src.preprocess.audio_features import MelSpectrogramResult, compute_log_mel_spectrogram

# Optional modules requiring cv2 / vision backends
try:
    from src.preprocess.augmentation import (
        AudioAugmentor,
        AugmentationRecord,
        VideoAugmentor,
        apply_modality_dropout,
    )
except ImportError:
    AudioAugmentor = None  # type: ignore[assignment, misc]
    AugmentationRecord = None  # type: ignore[assignment, misc]
    VideoAugmentor = None  # type: ignore[assignment, misc]
    apply_modality_dropout = None  # type: ignore[assignment, misc]

try:
    from src.preprocess.face_extraction import (
        FaceCropResult,
        FaceExtractor,
        WindowFaceResult,
    )
except ImportError:
    FaceCropResult = None  # type: ignore[assignment, misc]
    FaceExtractor = None  # type: ignore[assignment, misc]
    WindowFaceResult = None  # type: ignore[assignment, misc]

try:
    from src.preprocess.quality_features import (
        AudioQualityFeatures,
        VisualQualityFeatures,
        compute_audio_quality,
        compute_visual_quality,
    )
except ImportError:
    AudioQualityFeatures = None  # type: ignore[assignment, misc]
    VisualQualityFeatures = None  # type: ignore[assignment, misc]
    compute_audio_quality = None  # type: ignore[assignment, misc]
    compute_visual_quality = None  # type: ignore[assignment, misc]

__all__ = [
    "AlignmentResult",
    "AudioAugmentor",
    "AudioQualityFeatures",
    "AugmentationRecord",
    "FaceCropResult",
    "FaceExtractor",
    "MelSpectrogramResult",
    "TemporalAligner",
    "VideoAugmentor",
    "VisualQualityFeatures",
    "WindowFaceResult",
    "apply_modality_dropout",
    "compute_audio_quality",
    "compute_log_mel_spectrogram",
    "compute_visual_quality",
]
