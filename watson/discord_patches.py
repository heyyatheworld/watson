"""Discord/py-cord compatibility helpers: optional IPv4-only DNS and voice patches."""

from __future__ import annotations

import logging
import os
import socket
import time

_LOG = logging.getLogger(__name__)

_OPUS_FALLBACK_PATHS = [
    "/opt/homebrew/lib/libopus.dylib",
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",
    "libopus.so.0",
]


def maybe_apply_ipv4_resolver_patch() -> None:
    """
    When WATSON_FORCE_IPV4 is set, restrict socket name resolution to IPv4.

    Helps on hosts where IPv6 routes break Discord voice; disabled by default so
    other outbound connections keep normal dual-stack behaviour.
    """
    raw = os.getenv("WATSON_FORCE_IPV4", "").strip().lower()
    if raw not in ("1", "true", "yes"):
        return

    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return _orig_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = _ipv4_only_getaddrinfo
    _LOG.info("socket.getaddrinfo restricted to IPv4 (WATSON_FORCE_IPV4)")


def patch_voice_client_shutdown(logger: logging.Logger) -> None:
    """
    Avoid 'Task exception was never retrieved' on shutdown: when the bot closes,
    VoiceClient.disconnect() runs cleanup() while poll_voice_ws() may still be
    running. The library can set self.ws to MISSING, so the next
    self.ws.poll_event() raises AttributeError. Patch poll_voice_ws to exit
    cleanly in that case.
    """
    import discord

    try:
        VoiceClient = discord.voice_client.VoiceClient
        _orig_poll_voice_ws = VoiceClient.poll_voice_ws

        async def _patched_poll_voice_ws(self, reconnect: bool) -> None:
            try:
                await _orig_poll_voice_ws(self, reconnect)
            except AttributeError as e:
                if "poll_event" in str(e):
                    return
                raise

        VoiceClient.poll_voice_ws = _patched_poll_voice_ws
        logger.debug("VoiceClient.poll_voice_ws shutdown patch applied")
    except Exception as e:
        logger.warning("Could not patch VoiceClient for clean shutdown: %s", e)


def load_opus(logger: logging.Logger) -> None:
    """Load Opus library from OPUS_LIB_PATH or fallback paths. Required for voice."""
    import discord

    explicit = os.getenv("OPUS_LIB_PATH")
    paths = [explicit] if explicit else list(_OPUS_FALLBACK_PATHS)
    for path in paths:
        try:
            discord.opus.load_opus(path)
            logger.info("Opus loaded: %s", path)
            return
        except Exception as e:
            logger.debug("Opus load failed for %s: %s", path, e)
    logger.warning(
        "Opus could not be loaded from any path. Voice may fail with decode errors. "
        "Set OPUS_LIB_PATH to your libopus path (e.g. /opt/homebrew/lib/libopus.dylib on macOS)."
    )


def setup_discord_voice_audio_logging() -> None:
    """Reduce noisy voice decoder logs from discord/opus."""
    logging.getLogger("discord.voicereader").setLevel(logging.ERROR)
    logging.getLogger("discord.voicereader").propagate = False
    logging.getLogger("discord.opus").setLevel(logging.ERROR)


class SuppressOpusDecodeFilter(logging.Filter):
    """Suppress repeated 'Error occurred while decoding opus frame' log records."""

    _last_log_time = 0.0
    _cooldown_sec = 60.0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(getattr(record, "msg", "")) + str(getattr(record, "args", ()))
        if "decoding opus frame" not in msg.lower():
            return True
        now = time.monotonic()
        if now - SuppressOpusDecodeFilter._last_log_time >= SuppressOpusDecodeFilter._cooldown_sec:
            SuppressOpusDecodeFilter._last_log_time = now
            return True
        return False


def setup_discord_voice_patches(logger: logging.Logger) -> None:
    """Apply VoiceClient shutdown patch, load Opus, tune discord voice loggers."""
    patch_voice_client_shutdown(logger)
    load_opus(logger)
    setup_discord_voice_audio_logging()
