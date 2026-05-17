"""faster-whisper wrapper and transcript formatting."""

from __future__ import annotations

import logging
import threading

from faster_whisper import WhisperModel

from watson.config import SETTINGS

logger = logging.getLogger(__name__)

_whisper_model_instance: WhisperModel | None = None
_whisper_model_lock = threading.Lock()


def get_whisper_model() -> WhisperModel:
    """Load faster-whisper once on first transcription (avoids heavy import-time startup)."""
    global _whisper_model_instance
    if _whisper_model_instance is None:
        with _whisper_model_lock:
            if _whisper_model_instance is None:
                logger.info("Loading Whisper model (%s)...", SETTINGS.whisper_model)
                _whisper_model_instance = WhisperModel(
                    SETTINGS.whisper_model,
                    device=SETTINGS.whisper_device,
                    compute_type=SETTINGS.whisper_compute_type,
                )
                logger.info("Whisper ready")
    return _whisper_model_instance


def build_transcript_lines(phrases: list[dict]) -> str:
    """Build raw transcript text from list of {time, user, text}; used by once_done."""
    lines = []
    for p in phrases:
        m, s = divmod(int(p["time"]), 60)
        lines.append(f"[{m:02d}:{s:02d}] **{p['user']}**: {p['text']}\n")
    return "".join(lines)
