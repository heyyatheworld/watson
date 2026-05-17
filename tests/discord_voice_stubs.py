"""Discord-like stubs for voice tests."""

from __future__ import annotations


class FakeVoiceConnectionClosed(Exception):
    """Like ``discord.errors.ConnectionClosed``; code ``4017`` usually means DAVE E2EE rejected the handshake."""

    def __init__(self, code: int = 4017):
        self.code = code
        super().__init__(f"Connection closed ({code})")
