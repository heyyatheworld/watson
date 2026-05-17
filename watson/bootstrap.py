"""Configure logging, directories, env validation, and run the Discord bot."""

from __future__ import annotations

import logging
import os
import sys

from watson.config import SETTINGS
from watson.discord_patches import (
    SuppressOpusDecodeFilter,
    setup_discord_voice_patches,
)


class _SuppressUnclosedConnectionFilter(logging.Filter):
    """Suppress asyncio 'Unclosed connection' noise from py-cord voice retries."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Unclosed connection" in msg and "client_connection" in msg:
            return False
        return True


def configure_logging() -> logging.Logger:
    log_level = getattr(logging, SETTINGS.log_level_name.upper(), logging.INFO)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    if SETTINGS.log_file:
        fh = logging.FileHandler(SETTINGS.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(fh)
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.voice_client").setLevel(logging.DEBUG)
    logging.getLogger("asyncio").addFilter(_SuppressUnclosedConnectionFilter())
    logging.getLogger().addFilter(SuppressOpusDecodeFilter())
    return logging.getLogger("watson.bootstrap")


def run_bot() -> None:
    logger = configure_logging()
    os.makedirs(SETTINGS.temp_dir, exist_ok=True)
    os.makedirs(SETTINGS.recordings_dir, exist_ok=True)
    setup_discord_voice_patches(logger)
    logger.info("Temp dir (cleared after each transcription): %s", SETTINGS.temp_dir)

    if not SETTINGS.skip_env_check:
        from watson.env_check import check_environment

        check_environment(SETTINGS, logging.getLogger("watson.env_check"))

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error("DISCORD_TOKEN not found in .env")
        sys.exit("Set DISCORD_TOKEN in .env (see .env.example)")

    from watson.handlers import create_bot

    bot = create_bot()

    logger.info("Starting bot...")
    try:
        bot.run(token)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down.")
