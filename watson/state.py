"""Mutable process-global references for the Discord bot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discord.ext import commands

bot: commands.Bot | None = None

transcribing_guilds: set[int] = set()
