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


def test_record_rejects_same_guild_when_transcribing(main_module):
    """!record in the same guild is rejected while previous recording is transcribing."""
    main_module.transcribing_guilds.add(999)
    ctx = MagicMock()
    ctx.guild.id = 999
    ctx.voice_client = MagicMock()
    ctx.voice_client.recording = False
    ctx.send = AsyncMock(return_value=None)
    try:
        asyncio.run(main_module.record(ctx))
        call_args = ctx.send.call_args[0][0]
        assert "transcrib" in call_args.lower() or "wait" in call_args.lower()
    finally:
        main_module.transcribing_guilds.discard(999)


def test_record_allowed_other_guild_while_one_transcribing(main_module):
    """!record in another guild is allowed while guild A is transcribing (concurrent recordings)."""
    main_module.transcribing_guilds.add(111)
    ctx = MagicMock()
    ctx.guild.id = 222
    ctx.channel.name = "general"
    ctx.author = MagicMock()
    ctx.voice_client = MagicMock()
    ctx.voice_client.recording = False
    ctx.voice_client.channel = MagicMock()
    ctx.voice_client.channel.name = "voice"
    ctx.send = AsyncMock(return_value=None)
    ctx.voice_client.start_recording = MagicMock()
    try:
        asyncio.run(main_module.record(ctx))
        ctx.send.assert_called_once()
        call_args = ctx.send.call_args[0][0]
        assert "recording started" in call_args.lower() or "⏺" in call_args
        ctx.voice_client.start_recording.assert_called_once()
    finally:
        main_module.transcribing_guilds.discard(111)
