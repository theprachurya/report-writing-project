"""Raw media decoding subpackage.

Provides deterministic video and audio decoding at fixed rates for the
preprocessing pipeline. See AGENTS.md for build order.
"""

from src.decode.audio_decoder import (
    AudioDecodeResult,
    decode_audio,
    decode_audio_window,
)
from src.decode.video_decoder import (
    VideoDecodeResult,
    decode_video,
    decode_video_window,
)

__all__ = [
    "AudioDecodeResult",
    "VideoDecodeResult",
    "decode_audio",
    "decode_audio_window",
    "decode_video",
    "decode_video_window",
]
