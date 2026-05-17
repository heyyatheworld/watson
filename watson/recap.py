"""Ollama recap generation (blocking; call via asyncio.to_thread)."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from watson.config import Settings


def get_recap_sync(transcript: str, settings: Settings, logger: logging.Logger) -> str | None:
    """
    Generate a short recap via Ollama with retries.
    Returns None if disabled, prompt file missing, or on error after retries.
    """
    if not settings.ollama_recap_model:
        return None
    if not os.path.isfile(settings.recap_prompt_file):
        logger.warning("Recap prompt file not found: %s", settings.recap_prompt_file)
        return None
    try:
        with open(settings.recap_prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except OSError as e:
        logger.warning("Could not read recap prompt: %s", e)
        return None
    prompt = prompt_template.replace("{{TRANSCRIPT}}", transcript)
    if len(prompt) > 12000:
        prompt = prompt[:12000] + "\n\n[truncated]"
    client = ollama.Client(host=settings.ollama_host)
    last_error = None
    for attempt in range(settings.ollama_retries):
        try:
            response = client.chat(
                model=settings.ollama_recap_model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.get("message") or {}).get("content") or ""
            text = text.strip()
            if not text:
                return None
            cap = settings.recap_max_chars
            if len(text) > cap:
                text = text[: cap - 3].rstrip() + "..."
            return text
        except Exception as e:
            last_error = e
            if attempt < settings.ollama_retries - 1:
                logger.debug(
                    "Ollama recap attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt + 1,
                    settings.ollama_retries,
                    settings.ollama_retry_delay,
                    e,
                )
                time.sleep(settings.ollama_retry_delay)
    logger.warning(
        "Ollama recap failed after %d attempts: %s",
        settings.ollama_retries,
        last_error,
    )
    return None
