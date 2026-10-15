"""Decodes video files at fixed 25 fps and yields the frame sequence for downstream face detection.

Status: implemented — provides decode_video() for deterministic frame extraction at a
fixed target FPS, and decode_video_window() for extracting a 2-second / 16-frame window.
See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoDecodeResult:
    """Result of decoding a video file.

    Attributes
    ----------
    frames:
        Array of shape ``(N, H, W, 3)`` in RGB uint8.
    fps:
        The native fps of the source video.
    target_fps:
        The fps at which frames were sampled.
    total_native_frames:
        Total frames in the original video.
    timestamps_s:
        Per-frame timestamps in seconds.
    dropped_frame_indices:
        Indices of frames that could not be read (returned as black).
    """
    frames: np.ndarray                                # (N, H, W, 3) RGB uint8
    fps: float
    target_fps: int
    total_native_frames: int
    timestamps_s: np.ndarray                          # (N,) float64
    dropped_frame_indices: list[int] = field(default_factory=list)


def decode_video(
    video_path: str | Path,
    target_fps: int = 25,
) -> VideoDecodeResult:
    """Decode a video file at a fixed target FPS.

    Reads every frame and sub-samples to approximate *target_fps* uniformly.
    Frames are converted from BGR (OpenCV default) to RGB.

    Parameters
    ----------
    video_path:
        Path to the video file (supports container files or .npz fixtures).
    target_fps:
        Desired output frame rate.  Per Section 6, default is 25.

    Returns
    -------
    VideoDecodeResult
        Decoded frames and metadata.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If the video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Handle .npz video fixtures (portable for headless testing without media codecs)
    if video_path.suffix.lower() == ".npz":
        data = np.load(str(video_path))
        raw_frames = data["frames"]
        native_fps = float(data["fps"]) if "fps" in data else 25.0
        total_native = len(raw_frames)
        duration_s = total_native / native_fps
        n_output = max(1, int(round(duration_s * target_fps)))
        sample_indices = np.round(np.linspace(0, total_native - 1, n_output)).astype(int)
        sampled_frames = raw_frames[sample_indices]
        timestamps = sample_indices / native_fps
        return VideoDecodeResult(
            frames=sampled_frames,
            fps=native_fps,
            target_fps=target_fps,
            total_native_frames=total_native,
            timestamps_s=timestamps,
            dropped_frame_indices=[],
        )

    if cv2 is None:
        raise RuntimeError(
            f"OpenCV is required to decode {video_path.suffix} video containers. "
            "Please install opencv-python or use .npz video fixtures."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_native <= 0 or native_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video metadata: fps={native_fps}, frames={total_native}"
        )

    # Compute which native frame indices to sample for target_fps
    duration_s = total_native / native_fps
    n_output = max(1, int(round(duration_s * target_fps)))
    # Uniform sampling: map output index → native frame index
    sample_indices = np.round(
        np.linspace(0, total_native - 1, n_output)
    ).astype(int)

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    dropped: list[int] = []

    current_native_idx = 0
    for out_idx, target_native in enumerate(sample_indices):
        # Seek if needed (sequential reads are faster for small skips)
        if target_native != current_native_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_native)
            current_native_idx = target_native

        ret, frame = cap.read()
        current_native_idx += 1

        if ret and frame is not None:
            # BGR → RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # Failed read — insert a black frame and record the drop
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            dropped.append(out_idx)

        timestamps.append(target_native / native_fps)

    cap.release()

    result = VideoDecodeResult(
        frames=np.stack(frames, axis=0),
        fps=native_fps,
        target_fps=target_fps,
        total_native_frames=total_native,
        timestamps_s=np.array(timestamps, dtype=np.float64),
        dropped_frame_indices=dropped,
    )

    if dropped:
        logger.warning(
            "Dropped %d/%d frames in %s", len(dropped), n_output, video_path.name
        )

    return result


def decode_video_window(
    video_path: str | Path,
    start_s: float = 0.0,
    window_duration_s: float = 2.0,
    frames_per_window: int = 16,
    target_fps: int = 25,
) -> VideoDecodeResult:
    """Decode a 2-second window of *frames_per_window* uniformly spaced frames.

    This matches the input contract (Section 6): 2 s window, 16 frames at
    effective 8 fps from a 25 fps decode.

    Parameters
    ----------
    video_path:
        Path to the video file.
    start_s:
        Start time of the window in seconds.
    window_duration_s:
        Window duration in seconds (default 2.0).
    frames_per_window:
        Number of frames to extract per window (default 16).
    target_fps:
        Base decode frame rate (default 25).

    Returns
    -------
    VideoDecodeResult
        The 16 extracted frames and metadata.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if start_s < 0 or window_duration_s <= 0 or frames_per_window <= 0:
        raise ValueError("start_s must be non-negative and window/frame counts must be positive")

    # Decode/resample the source at the configured fixed FPS first, then sample
    # the requested window. This keeps container and fixture behavior aligned.
    full_res = decode_video(video_path, target_fps=target_fps)
    if len(full_res.frames) == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    frame_height, frame_width = full_res.frames.shape[1:3]
    duration_s = full_res.total_native_frames / max(full_res.fps, 1e-8)
    timestamps = start_s + np.arange(frames_per_window, dtype=np.float64) * (
        window_duration_s / frames_per_window
    )
    frames = np.zeros(
        (frames_per_window, frame_height, frame_width, 3), dtype=np.uint8
    )
    dropped: list[int] = []
    decode_drops = set(full_res.dropped_frame_indices)

    for out_idx, timestamp in enumerate(timestamps):
        if timestamp >= duration_s:
            dropped.append(out_idx)
            continue
        right = int(np.searchsorted(full_res.timestamps_s, timestamp, side="left"))
        right = min(right, len(full_res.timestamps_s) - 1)
        left = max(0, right - 1)
        source_idx = min(
            (left, right),
            key=lambda idx: abs(float(full_res.timestamps_s[idx]) - timestamp),
        )
        if source_idx in decode_drops:
            dropped.append(out_idx)
            continue
        frames[out_idx] = full_res.frames[source_idx]

    return VideoDecodeResult(
        frames=frames,
        fps=full_res.fps,
        target_fps=target_fps,
        total_native_frames=full_res.total_native_frames,
        timestamps_s=timestamps,
        dropped_frame_indices=dropped,
    )
