"""Permission checks for sensitive voice commands."""

from __future__ import annotations

from discord.ext import commands

from watson.config import SETTINGS


async def ensure_voice_operator(ctx: commands.Context) -> bool:
    """
    When lockdown is enabled, allow only Manage Server, administrator, or roles in
    WATSON_ALLOWED_ROLE_IDS. Public servers should set lockdown in production.
    """
    if ctx.guild is None:
        await ctx.send("Use Watson voice commands from a server text channel.")
        return False
    if not SETTINGS.lockdown_voice_commands:
        return True
    member = ctx.author
    if SETTINGS.allowed_role_ids and any(
        r.id in SETTINGS.allowed_role_ids for r in member.roles
    ):
        return True
    perms = member.guild_permissions
    if perms.manage_guild or perms.administrator:
        return True
    await ctx.send(
        "🔒 Voice commands are locked down. You need **Manage Server**, "
        "**Administrator**, or a role ID listed in `WATSON_ALLOWED_ROLE_IDS`. "
        "Unset `WATSON_LOCKDOWN_VOICE_COMMANDS` for trusted/private servers."
    )
    return False
