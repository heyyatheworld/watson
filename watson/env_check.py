"""Startup checks before connecting to Discord."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from watson.config import Settings


def check_environment(settings: Settings, logger: logging.Logger) -> None:
    """
    Verify temp and recordings dirs are writable; if recap is enabled, verify Ollama is reachable.
    Exits with a clear message on failure.
    """
    for name, path in [
        ("WATSON_TEMP_DIR", settings.temp_dir),
        ("WATSON_RECORDINGS_DIR", settings.recordings_dir),
    ]:
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".watson_write_test")
            with open(test_file, "w") as f:
                f.write("")
            os.remove(test_file)
        except OSError as e:
            logger.error("%s is not writable: %s — %s", name, path, e)
            sys.exit(1)
    if settings.ollama_recap_model:
        if not os.path.isfile(settings.recap_prompt_file):
            logger.error(
                "OLLAMA_RECAP_MODEL is set but recap prompt file not found: %s",
                settings.recap_prompt_file,
            )
            sys.exit(1)
        try:
            client = ollama.Client(host=settings.ollama_host)
            client.list()
        except Exception as e:
            logger.error(
                "Ollama is not reachable at %s (OLLAMA_RECAP_MODEL=%s): %s",
                settings.ollama_host,
                settings.ollama_recap_model,
                e,
            )
            sys.exit(1)
    logger.debug("Environment check passed")
