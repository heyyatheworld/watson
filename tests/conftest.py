"""
Pytest fixtures: fake discord, faster_whisper, ollama, psutil so main can be
imported without real connections or model load.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def main_module():
    """
    Import main with Discord, faster_whisper, ollama and psutil mocked.
    No token or model required.
    """
    fake_discord = MagicMock()
    fake_discord.opus.load_opus = MagicMock()
    fake_ext = MagicMock()
    fake_commands = MagicMock()
    fake_bot = MagicMock()
    fake_bot.run = MagicMock()
    fake_bot.command = lambda: (lambda fn: fn)  # @bot.command() returns decorator that returns fn
    fake_bot.event = lambda fn: fn  # @bot.event returns the function unchanged
    fake_commands.Bot = MagicMock(return_value=fake_bot)
    fake_ext.commands = fake_commands
    fake_discord.ext = fake_ext
    fake_fw = MagicMock()
    fake_psutil = MagicMock()
    fake_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
    fake_ollama = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {
                "discord": fake_discord,
                "discord.ext": fake_ext,
                "discord.ext.commands": fake_commands,
                "faster_whisper": fake_fw,
                "ollama": fake_ollama,
                "psutil": fake_psutil,
            },
        ),
        patch.dict(
            os.environ,
            {
                "DISCORD_TOKEN": "test-token",
                "LOG_LEVEL": "WARNING",
                "WATSON_SKIP_ENV_CHECK": "1",
            },
            clear=False,
        ),
        patch("dotenv.load_dotenv", MagicMock()),
        patch("tempfile.gettempdir", return_value="/tmp"),
        patch("os.makedirs", MagicMock()),
    ):
        import main as _main
        _main.bot = MagicMock()
        return _main
