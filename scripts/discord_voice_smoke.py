#!/usr/bin/env python3
"""
Minimal Discord voice connectivity diagnostic (standalone).

Uses aggressive IPv4-only DNS and optional VoiceClient cipher narrowing — for
narrowing down UDP/gateway issues only. Prefer running the full Watson bot via
`python main.py`; use `WATSON_FORCE_IPV4` there when appropriate.

Requires: discord.py/py-cord, PyNaCl, Opus (see README).
"""

import asyncio
import os
import platform
import socket

import discord
from discord.ext import commands
from dotenv import load_dotenv

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo

print("=== Watson voice diagnostic ===")
print(f"Python: {platform.python_version()}, OS: {platform.system()}")

try:
    import nacl

    print(f"PyNaCl: OK ({nacl.__version__})")
except ImportError:
    print("PyNaCl: missing (voice encryption will fail)")

if not discord.opus.is_loaded():
    opus_path = "/opt/homebrew/lib/libopus.dylib"
    try:
        discord.opus.load_opus(opus_path)
        print(f"Opus: loaded {opus_path}")
    except Exception:
        print("Opus: failed to load (install opus / set OPUS_LIB_PATH)")
else:
    print("Opus: already loaded")

load_dotenv()
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Bot online: {bot.user.name} ({bot.user.id}) — try !join from voice")


@bot.command()
async def join(ctx):
    if not ctx.author.voice:
        return await ctx.send("Join a voice channel first.")

    channel = ctx.author.voice.channel
    print(f"Connecting to {channel.name} ({channel.id}), rtc_region={channel.rtc_region!r}")

    if ctx.voice_client:
        await ctx.voice_client.disconnect(force=True)
        await asyncio.sleep(2)

    try:
        await asyncio.wait_for(channel.connect(timeout=20, reconnect=False), timeout=25)
        await ctx.send("Connected.")
        vc = ctx.voice_client
        if vc and vc.ws:
            print(f"Voice WS OK, mode={vc.mode!r}")
    except asyncio.TimeoutError:
        print("Timeout waiting for voice connection")
    except discord.errors.ConnectionClosed as e:
        print(f"ConnectionClosed code={e.code}")
        if e.code == 4006:
            print("Hint: invalid session — reset bot token in Developer Portal")
        elif e.code == 4017:
            print("Hint: UDP / encryption — check PyNaCl, firewall, try WATSON_FORCE_IPV4 on main bot")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise


token = os.getenv("DISCORD_TOKEN")
if not token:
    print("DISCORD_TOKEN missing (.env)")
else:
    try:
        from discord.voice_client import VoiceClient

        VoiceClient.supported_modes = ("xsalsa20_poly1305",)
        bot.run(token)
    except Exception as e:
        print(f"Startup failed: {e}")
