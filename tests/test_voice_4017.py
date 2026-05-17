"""Discord voice close code handling (4017 = DAVE E2EE handshake rejection)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from tests.discord_voice_stubs import FakeVoiceConnectionClosed


def test_join_handles_discord_voice_4017_connection_closed(main_module):
    """
    Discord may close voice with ``4017`` when mandatory **DAVE** (E2EE) validation fails
    (gateway rejects bots whose library cannot complete the handshake). ``!join`` must swallow
    the error and notify the channel like any other ``ConnectionClosed``.
    """
    async def boom_connect(timeout=90.0, reconnect=True):  # noqa: ARG001
        raise FakeVoiceConnectionClosed(4017)

    ch = MagicMock()
    ch.name = "test-vc"
    ch.id = 999
    ch.connect = boom_connect

    guild = MagicMock()
    guild.id = 4242
    guild.voice_client = None

    ctx = MagicMock()
    ctx.guild = guild
    ctx.voice_client = None
    ctx.author.voice = MagicMock()
    ctx.author.voice.channel = ch
    ctx.send = AsyncMock(return_value=None)

    asyncio.run(main_module.join(ctx))

    ctx.send.assert_called_once()
    text = ctx.send.call_args[0][0].lower()
    assert "could not connect" in text or "voice" in text
    assert "connectionclosed" in text or "details" in text
