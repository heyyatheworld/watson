"""Typed configuration loaded from the environment (after dotenv in entrypoint)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _truthy(name: str) -> bool:
    v = os.getenv(name)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes")


def _comma_role_ids(name: str) -> frozenset[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return frozenset()
    parts = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(int(chunk))
    return frozenset(parts)


@dataclass(frozen=True)
class Settings:
    temp_dir: str
    recordings_dir: str
    log_level_name: str
    log_file: str | None
    bot_prefix: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    transcript_language: str | None
    transcript_beam_size: int
    transcript_junk_phrases: tuple[str, ...]
    max_recording_minutes: int
    warning_before_stop_minutes: int
    ollama_recap_model: str | None
    recap_prompt_file: str
    recap_max_chars: int
    ollama_retries: int
    ollama_retry_delay: float
    ollama_host: str
    skip_env_check: bool
    lockdown_voice_commands: bool
    allowed_role_ids: frozenset[int]
    discord_hide_paths: bool

    @property
    def max_recording_seconds(self) -> int:
        return self.max_recording_minutes * 60

    @property
    def warning_before_stop_seconds(self) -> int:
        return self.warning_before_stop_minutes * 60


def load_settings() -> Settings:
    default_junk = "editor|subtitles|thanks for watching|to be continued"
    junk_raw = os.getenv("TRANSCRIPT_JUNK_PHRASES", default_junk)
    junk = tuple(p.strip() for p in junk_raw.split("|") if p.strip())
    recap_default = _REPO_ROOT / "prompts" / "recap.txt"
    recap_file = os.getenv("RECAP_PROMPT_FILE") or str(recap_default)
    tl = (os.getenv("TRANSCRIPT_LANGUAGE") or "").strip()
    om = (os.getenv("OLLAMA_RECAP_MODEL") or "").strip()
    return Settings(
        temp_dir=os.getenv("WATSON_TEMP_DIR") or "./temp",
        recordings_dir=os.getenv("WATSON_RECORDINGS_DIR") or "./recordings",
        log_level_name=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE") or None,
        bot_prefix=os.getenv("BOT_COMMAND_PREFIX", "!"),
        whisper_model=os.getenv("WHISPER_MODEL", "turbo"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        transcript_language=tl or None,
        transcript_beam_size=int(os.getenv("TRANSCRIPT_BEAM_SIZE", "5")),
        transcript_junk_phrases=junk,
        max_recording_minutes=int(os.getenv("RECORDING_MAX_MINUTES", "30")),
        warning_before_stop_minutes=int(os.getenv("WARNING_BEFORE_STOP_MINUTES", "5")),
        ollama_recap_model=om or None,
        recap_prompt_file=recap_file,
        recap_max_chars=400,
        ollama_retries=3,
        ollama_retry_delay=2.0,
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        skip_env_check=_truthy("WATSON_SKIP_ENV_CHECK"),
        lockdown_voice_commands=_truthy("WATSON_LOCKDOWN_VOICE_COMMANDS"),
        allowed_role_ids=_comma_role_ids("WATSON_ALLOWED_ROLE_IDS"),
        discord_hide_paths=_truthy("WATSON_DISCORD_HIDE_PATHS"),
    )


SETTINGS = load_settings()
