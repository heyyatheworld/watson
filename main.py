"""
Discord bot: record voice channel audio, transcribe with faster-whisper,
optionally summarize with Ollama. Saves WAV and transcript to disk; posts recap and file links.
"""

from dotenv import load_dotenv

load_dotenv()

from watson.discord_patches import maybe_apply_ipv4_resolver_patch

maybe_apply_ipv4_resolver_patch()

from watson.bootstrap import run_bot

if __name__ == "__main__":
    run_bot()
