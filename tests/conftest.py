"""
Pytest fixtures: fake discord, faster_whisper, ollama so bot logic can import
without real connections or loading Whisper weights.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.discord_voice_stubs import FakeVoiceConnectionClosed


@pytest.fixture(scope="module")
def main_module():
    """
    watson.handlers + watson.transcribe with mocked Discord stack.
    Reloads config after patching env so SETTINGS reflects test defaults.
    """
    fake_discord = MagicMock()
    fake_discord.opus.load_opus = MagicMock()
    fake_discord.errors.ConnectionClosed = FakeVoiceConnectionClosed
    fake_discord.sinks.WaveSink = MagicMock
    fake_ext = MagicMock()
    fake_commands = MagicMock()
    fake_bot = MagicMock()
    fake_bot.run = MagicMock()
    fake_bot.command = lambda: (lambda fn: fn)
    fake_bot.event = lambda fn: fn
    fake_commands.Bot = MagicMock(return_value=fake_bot)
    fake_ext.commands = fake_commands
    fake_discord.ext = fake_ext
    fake_fw = MagicMock()
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
        patch("tempfile.gettempdir", return_value="/tmp"),
        patch("os.makedirs", MagicMock()),
    ):
        import importlib

        import watson.config as cfg
        import watson.handlers as handlers
        import watson.permissions as perm
        import watson.state as st
        import watson.transcribe as tr

        importlib.reload(cfg)
        importlib.reload(perm)
        importlib.reload(tr)
        importlib.reload(handlers)

        st.bot = MagicMock()
        st.transcribing_guilds.clear()

        return SimpleNamespace(
            build_transcript_lines=tr.build_transcript_lines,
            join=handlers.join,
            record=handlers.record,
            transcribing_guilds=st.transcribing_guilds,
            MAX_RECORDING_MINUTES=cfg.SETTINGS.max_recording_minutes,
            MAX_RECORDING_SECONDS=cfg.SETTINGS.max_recording_seconds,
        )
