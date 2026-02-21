"""Basic tests for Watson bot logic (helpers, config, commands with mocks)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_build_transcript_lines_empty(main_module):
    """Empty phrases list yields empty string."""
    assert main_module.build_transcript_lines([]) == ""


def test_build_transcript_lines_single(main_module):
    """Single phrase is formatted with timestamp and user."""
    phrases = [{"time": 0, "user": "Alice", "text": "Hello"}]
    assert main_module.build_transcript_lines(phrases) == "[00:00] **Alice**: Hello\n"


def test_build_transcript_lines_multiple(main_module):
    """Multiple phrases are formatted with timestamps."""
    phrases = [
        {"time": 65, "user": "Bob", "text": "Minute one"},
        {"time": 0, "user": "Alice", "text": "Start"},
    ]
    out = main_module.build_transcript_lines(phrases)
    assert "[00:00] **Alice**: Start\n" in out
    assert "[01:05] **Bob**: Minute one\n" in out


def test_memory_mb_returns_float_or_none(main_module):
    """_memory_mb returns a non-negative float or None."""
    result = main_module._memory_mb()
    assert result is None or (isinstance(result, float) and result >= 0)


def test_log_memory_does_not_raise(main_module):
    """_log_memory does not raise for any stage name."""
    main_module._log_memory("test_stage")


def test_recording_limit_config(main_module):
    """MAX_RECORDING_SECONDS is 30 * 60 when env is default."""
    # conftest does not set RECORDING_MAX_MINUTES, so default 30 is used
    assert main_module.MAX_RECORDING_MINUTES == 30
    assert main_module.MAX_RECORDING_SECONDS == 30 * 60


def test_record_rejects_when_not_in_voice(main_module):
    """!record sends invite message when bot is not in a voice channel."""
    ctx = MagicMock()
    ctx.voice_client = None
    ctx.send = AsyncMock(return_value=None)

    asyncio.run(main_module.record(ctx))

    ctx.send.assert_called_once()
    call_args = ctx.send.call_args[0][0]
    assert "join" in call_args.lower() or "invite" in call_args.lower()
