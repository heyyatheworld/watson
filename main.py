"""
Discord bot: record voice channel audio, transcribe with faster-whisper,
optionally summarize with Ollama. Saves WAV and transcript to disk; posts recap and file links.
"""

import asyncio
import gc
import logging
import os
import signal
import shutil
import sys
import tempfile
import time
from datetime import datetime

import discord
import ollama
import psutil
from discord.ext import commands
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

_watson_temp_dir = os.getenv("WATSON_TEMP_DIR") or "./temp"
os.makedirs(_watson_temp_dir, exist_ok=True)

_watson_recordings_dir = os.getenv("WATSON_RECORDINGS_DIR") or "./recordings"
os.makedirs(_watson_recordings_dir, exist_ok=True)

log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
log_file = os.getenv("LOG_FILE")
if log_file:
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

logger = logging.getLogger(__name__)
logging.getLogger("discord").setLevel(logging.WARNING)


def _memory_mb():
    """Return current process RSS in MB, or None if unavailable."""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _log_memory(stage: str):
    """Log diagnostic memory usage at the given stage."""
    mb = _memory_mb()
    if mb is not None:
        logger.info("Memory [%s]: %.1f MB RSS", stage, mb)


def build_transcript_lines(phrases: list[dict]) -> str:
    """Build raw transcript text from list of {time, user, text}; used by once_done."""
    lines = []
    for p in phrases:
        m, s = divmod(int(p["time"]), 60)
        lines.append(f"[{m:02d}:{s:02d}] **{p['user']}**: {p['text']}\n")
    return "".join(lines)


logging.getLogger("discord.voicereader").setLevel(logging.ERROR)
logging.getLogger("discord.voicereader").propagate = False
logging.getLogger("discord.opus").setLevel(logging.ERROR)

_OPUS_FALLBACK_PATHS = [
    "/opt/homebrew/lib/libopus.dylib",
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",
    "libopus.so.0",
]


def _load_opus() -> None:
    """Load Opus library from OPUS_LIB_PATH or fallback paths. Required for voice."""
    explicit = os.getenv("OPUS_LIB_PATH")
    if explicit:
        paths = [explicit]
    else:
        paths = _OPUS_FALLBACK_PATHS
    for path in paths:
        try:
            discord.opus.load_opus(path)
            logger.info("Opus loaded: %s", path)
            return
        except Exception as e:
            logger.debug("Opus load failed for %s: %s", path, e)
    logger.warning(
        "Opus could not be loaded from any path. Voice may fail with decode errors. "
        "Set OPUS_LIB_PATH to your libopus path (e.g. /opt/homebrew/lib/libopus.dylib on macOS)."
    )


_load_opus()
_log_memory("after_opus")

_whisper_model = os.getenv("WHISPER_MODEL", "turbo")
_whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
_whisper_compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
logger.info("Loading Whisper model (%s)...", _whisper_model)
model = WhisperModel(_whisper_model, device=_whisper_device, compute_type=_whisper_compute)
logger.info("Whisper ready")
logger.info("Temp dir (cleared after each transcription): %s", _watson_temp_dir)
_log_memory("after_whisper_load")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

_bot_prefix = os.getenv("BOT_COMMAND_PREFIX", "!")
bot: commands.Bot | None = None

transcribing_guilds = set()

MAX_RECORDING_MINUTES = int(os.getenv("RECORDING_MAX_MINUTES", "30"))
MAX_RECORDING_SECONDS = MAX_RECORDING_MINUTES * 60
WARNING_BEFORE_STOP_MINUTES = int(os.getenv("WARNING_BEFORE_STOP_MINUTES", "5"))
WARNING_BEFORE_STOP_SECONDS = WARNING_BEFORE_STOP_MINUTES * 60

_transcript_lang = (os.getenv("TRANSCRIPT_LANGUAGE") or "").strip()
TRANSCRIPT_LANGUAGE = _transcript_lang or None
TRANSCRIPT_BEAM_SIZE = int(os.getenv("TRANSCRIPT_BEAM_SIZE", "5"))

_default_junk = "editor|subtitles|thanks for watching|to be continued"
TRANSCRIPT_JUNK_PHRASES = [
    p.strip() for p in os.getenv("TRANSCRIPT_JUNK_PHRASES", _default_junk).split("|") if p.strip()
]

OLLAMA_RECAP_MODEL = (os.getenv("OLLAMA_RECAP_MODEL") or "").strip() or None
_recap_prompt_file = os.getenv("RECAP_PROMPT_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "recap.txt"
)
RECAP_MAX_CHARS = 400

OLLAMA_RETRIES = 3
OLLAMA_RETRY_DELAY = 2.0


def _check_environment() -> None:
    """
    Verify temp and recordings dirs are writable; if recap is enabled, verify Ollama is reachable.
    Exits with a clear message on failure.
    """
    for name, path in [
        ("WATSON_TEMP_DIR", _watson_temp_dir),
        ("WATSON_RECORDINGS_DIR", _watson_recordings_dir),
    ]:
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".watson_write_test")
            with open(test_file, "w") as f:
                f.write("")
            os.remove(test_file)
        except OSError as e:
            logger.error("%s is not writable: %s — %s", name, path, e)
            sys.exit(1)
    if OLLAMA_RECAP_MODEL:
        if not os.path.isfile(_recap_prompt_file):
            logger.error(
                "OLLAMA_RECAP_MODEL is set but recap prompt file not found: %s",
                _recap_prompt_file,
            )
            sys.exit(1)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            client = ollama.Client(host=ollama_host)
            client.list()
        except Exception as e:
            logger.error(
                "Ollama is not reachable at %s (OLLAMA_RECAP_MODEL=%s): %s",
                ollama_host,
                OLLAMA_RECAP_MODEL,
                e,
            )
            sys.exit(1)
    logger.debug("Environment check passed")


if not os.getenv("WATSON_SKIP_ENV_CHECK"):
    _check_environment()


async def on_ready() -> None:
    """Log bot name, ID, and guild count when the bot comes online."""
    assert bot is not None
    logger.info(
        "Watson online — %s (ID: %s), guilds: %d",
        bot.user.name,
        bot.user.id,
        len(bot.guilds),
    )
    _log_memory("on_ready")
    for guild in bot.guilds:
        logger.info("  Guild: %s (ID: %s)", guild.name, guild.id)


async def on_voice_state_update(member, before, after) -> None:
    """
    When the last human leaves the bot's voice channel, stop recording and leave.
    Ignores mute/deafen (same channel); only reacts to actual leave.
    """
    if before.channel is None:
        return
    if after.channel is not None and after.channel.id == before.channel.id:
        return
    voice_client = member.guild.voice_client
    if not voice_client or voice_client.channel != before.channel:
        return
    humans_remaining = [m for m in before.channel.members if m != member and not m.bot]
    logger.debug(
        "Voice state: %s left %s (guild %s), humans remaining: %d",
        member.display_name,
        before.channel.name,
        member.guild.id,
        len(humans_remaining),
    )
    if len(humans_remaining) != 0:
        return
    if voice_client.recording:
        logger.info(
            "Channel %s (guild %s) empty, stopping recording → transcription will run",
            before.channel.name,
            member.guild.id,
        )
        voice_client.stop_recording()
    await voice_client.disconnect()
    logger.info(
        "Left voice channel %s (guild %s), no users left",
        before.channel.name,
        member.guild.id,
    )


async def check(ctx) -> None:
    """Reply with connection status and bot permissions in the current channel."""
    logger.info(
        "!check from %s in %s/#%s (guild %s)",
        ctx.author,
        ctx.guild.name,
        ctx.channel.name,
        ctx.guild.id,
    )
    perms = ctx.channel.permissions_for(ctx.me)
    status = [
        f"✅ **Connection:** OK",
        f"🎤 **Voice channel:** {'✅' if ctx.author.voice else '❌ (you are not in a channel)'}",
        f"📝 **Send messages:** {'✅' if perms.send_messages else '❌'}",
        f"📎 **Attach files:** {'✅' if perms.attach_files else '❌'}",
        f"📜 **Read history:** {'✅' if perms.read_message_history else '❌'}",
        f"🎙 **Speak:** {'✅' if perms.speak else '❌'}",
    ]
    embed = discord.Embed(
        title="Watson system check",
        description="\n".join(status),
        color=discord.Color.blue() if perms.attach_files else discord.Color.red(),
    )
    await ctx.send(embed=embed)


async def join(ctx) -> None:
    """Join the voice channel the author is in."""
    logger.info("!join from %s in guild %s", ctx.author, ctx.guild.id)
    if ctx.voice_client:
        logger.debug("Rejected: already in channel %s", ctx.voice_client.channel.name)
        return await ctx.send("I'm already in a channel! 🎙")
    if ctx.author.voice:
        ch = ctx.author.voice.channel
        await ch.connect()
        logger.info("Joined voice channel %s (guild %s)", ch.name, ctx.guild.id)
        await ctx.send("🎩 Joined. Ready.")
    else:
        logger.debug("Rejected: author not in voice channel")
        await ctx.send("Join a voice channel first.")


async def _enforce_recording_limit(guild_id: int, channel_id: int) -> None:
    """
    After MAX_RECORDING_SECONDS, stop the current recording and notify.
    Sends a warning to the channel WARNING_BEFORE_STOP_MINUTES before the limit.
    """
    assert bot is not None
    warning_after = max(0, MAX_RECORDING_SECONDS - WARNING_BEFORE_STOP_SECONDS)
    await asyncio.sleep(warning_after)
    guild = bot.get_guild(guild_id)
    if not guild:
        return
    ch = guild.get_channel(channel_id)
    voice = guild.voice_client
    if (
        voice
        and voice.recording
        and ch
        and WARNING_BEFORE_STOP_SECONDS < MAX_RECORDING_SECONDS
    ):
        try:
            await ch.send(
                f"⚠️ **{WARNING_BEFORE_STOP_MINUTES} min left** until auto-stop (limit {MAX_RECORDING_MINUTES} min). "
                "Stop manually with `!stop`."
            )
        except discord.DiscordException:
            pass
    remaining = MAX_RECORDING_SECONDS - warning_after
    await asyncio.sleep(remaining)
    guild = bot.get_guild(guild_id)
    if not guild:
        return
    voice = guild.voice_client
    if voice and voice.recording:
        logger.info(
            "Recording limit (%d min) reached for guild %s, stopping",
            MAX_RECORDING_MINUTES,
            guild_id,
        )
        voice.stop_recording()
        ch = guild.get_channel(channel_id)
        if ch:
            try:
                await ch.send(
                    f"⏱ **{MAX_RECORDING_MINUTES} min limit reached.** Use `!record` to start a new recording."
                )
            except discord.DiscordException:
                pass


async def record(ctx) -> None:
    """Start recording in the current voice channel (max length from RECORDING_MAX_MINUTES)."""
    logger.info(
        "!record from %s in guild %s (channel %s)",
        ctx.author,
        ctx.guild.id,
        ctx.channel.name,
    )
    voice = ctx.voice_client
    if not voice:
        logger.debug("Rejected: bot not in voice channel")
        return await ctx.send("Invite me with !join first.")
    if voice.recording:
        logger.debug("Rejected: recording already in progress")
        return await ctx.send("⚠️ Recording is already in progress.")
    if ctx.guild.id in transcribing_guilds:
        logger.debug("Rejected: transcription in progress for guild %s", ctx.guild.id)
        return await ctx.send(
            "⚠️ Previous recording is still being transcribed. Wait for it to finish."
        )

    logger.info(
        "Recording started in %s (guild %s), limit %d min",
        voice.channel.name,
        ctx.guild.id,
        MAX_RECORDING_MINUTES,
    )
    await ctx.send(f"⏺ **Recording started.** (max {MAX_RECORDING_MINUTES} min)")
    voice.start_recording(discord.sinks.WaveSink(), once_done, ctx.channel)
    asyncio.create_task(_enforce_recording_limit(ctx.guild.id, ctx.channel.id))


async def stop(ctx) -> None:
    """Stop the current recording and run transcription and recap."""
    logger.info("!stop from %s in guild %s", ctx.author, ctx.guild.id)
    voice = ctx.voice_client
    if voice and voice.recording:
        logger.info(
            "Stopping recording in %s (guild %s), transcript will follow",
            voice.channel.name,
            ctx.guild.id,
        )
        voice.stop_recording()
        await ctx.send("⏹ Recording stopped. Building transcript...")
    else:
        logger.debug("Rejected: not recording")
        await ctx.send("I'm not recording right now.")


async def leave(ctx) -> None:
    """Leave the current voice channel."""
    logger.info("!leave from %s in guild %s", ctx.author, ctx.guild.id)
    if ctx.voice_client:
        ch_name = ctx.voice_client.channel.name
        await ctx.voice_client.disconnect()
        logger.info("Left voice channel %s (guild %s)", ch_name, ctx.guild.id)
        await ctx.send("Bye!")
    else:
        logger.debug("Rejected: not in voice channel")
        await ctx.send("I'm not in a voice channel.")


def _get_recap_sync(transcript: str) -> str | None:
    """
    Generate a short recap via Ollama with retries.
    Returns None if disabled, prompt file missing, or on error after retries.
    """
    if not OLLAMA_RECAP_MODEL:
        return None
    if not os.path.isfile(_recap_prompt_file):
        logger.warning("Recap prompt file not found: %s", _recap_prompt_file)
        return None
    try:
        with open(_recap_prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except OSError as e:
        logger.warning("Could not read recap prompt: %s", e)
        return None
    prompt = prompt_template.replace("{{TRANSCRIPT}}", transcript)
    if len(prompt) > 12000:
        prompt = prompt[:12000] + "\n\n[truncated]"
    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    last_error = None
    for attempt in range(OLLAMA_RETRIES):
        try:
            response = client.chat(
                model=OLLAMA_RECAP_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.get("message") or {}).get("content") or ""
            text = text.strip()
            if not text:
                return None
            if len(text) > RECAP_MAX_CHARS:
                text = text[: RECAP_MAX_CHARS - 3].rstrip() + "..."
            return text
        except Exception as e:
            last_error = e
            if attempt < OLLAMA_RETRIES - 1:
                logger.debug(
                    "Ollama recap attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt + 1,
                    OLLAMA_RETRIES,
                    OLLAMA_RETRY_DELAY,
                    e,
                )
                time.sleep(OLLAMA_RETRY_DELAY)
    logger.warning("Ollama recap failed after %d attempts: %s", OLLAMA_RETRIES, last_error)
    return None


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args) -> None:
    """
    Process recorded audio: transcribe with Whisper, save WAV and transcript to recordings,
    post recap (if Ollama enabled) and file links to the channel.
    """
    assert bot is not None
    guild_id = channel.guild.id
    guild_name = channel.guild.name
    num_participants = len(sink.audio_data) if sink.audio_data else 0

    logger.info(
        "once_done: guild %s (%s), channel %s, participants: %d",
        guild_id,
        guild_name,
        channel.name,
        num_participants,
    )

    if not sink.audio_data:
        logger.info("Empty recording for guild %s, skipping transcription", guild_id)
        await channel.send("📭 Recording is empty (silence).")
        return

    transcribing_guilds.add(guild_id)
    logger.debug("Added guild %s to transcribing_guilds", guild_id)
    _log_memory("transcription_start")
    status_msg = await channel.send("⚙️ **Watson is processing audio...**")

    all_phrases = []
    junk_phrases = TRANSCRIPT_JUNK_PHRASES
    temp_files = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_guild = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in guild_name
    )
    safe_channel = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in channel.name
    )
    temp_guild_dir = os.path.join(_watson_temp_dir, str(guild_id))
    os.makedirs(temp_guild_dir, exist_ok=True)

    try:
        for user_id, audio in sink.audio_data.items():
            temp_path = os.path.join(temp_guild_dir, f"temp_{user_id}.wav")

            audio.file.seek(0)
            data = audio.file.read()
            data_len = len(data)

            if data_len < 2000:
                logger.debug(
                    "Skipping user %s: audio too short (%d bytes)", user_id, data_len
                )
                continue

            with open(temp_path, "wb") as f:
                f.write(data)
            logger.debug(
                "Saved to temp %s (%d bytes), user %s", temp_path, data_len, user_id
            )
            temp_files.append((temp_path, user_id))
            await asyncio.sleep(0)

            try:

                def _transcribe(path: str):
                    """Run Whisper on path; return list of segments."""
                    segments_iter, _ = model.transcribe(
                        path,
                        beam_size=TRANSCRIPT_BEAM_SIZE,
                        language=TRANSCRIPT_LANGUAGE,
                    )
                    return list(segments_iter)

                segments_list = await asyncio.to_thread(_transcribe, temp_path)
                await asyncio.sleep(0)
                num_segments = len(segments_list)
                logger.info("Transcribed user %s: %d segments", user_id, num_segments)
                _log_memory("after_transcribe_user_%s" % user_id)

                user_obj = bot.get_user(user_id)
                username = user_obj.display_name if user_obj else f"User {user_id}"

                for seg in segments_list:
                    text = (seg.text or "").strip()
                    if (
                        not any(junk in text.lower() for junk in junk_phrases)
                        and len(text) > 1
                    ):
                        all_phrases.append(
                            {"time": seg.start, "user": username, "text": text}
                        )
                del segments_list
            except Exception as e:
                logger.exception("Whisper error for user %s: %s", user_id, e)

        all_phrases.sort(key=lambda x: x["time"])
        logger.debug("Collected %d phrases", len(all_phrases))

        raw_transcript = build_transcript_lines(all_phrases)

        if not raw_transcript:
            logger.info("No speech recognized for guild %s", guild_id)
            await status_msg.edit(content="😶 Could not recognize any speech.")
            return

        transcript_plain = raw_transcript.replace("**", "")

        await asyncio.sleep(0)
        recap = None
        if OLLAMA_RECAP_MODEL:
            recap = await asyncio.to_thread(
                _get_recap_sync, transcript_plain
            )
            await asyncio.sleep(0)
        recap_block = (recap + "\n\n") if recap else ""

        transcript_saved_path = os.path.join(
            _watson_recordings_dir,
            f"{timestamp}-{safe_guild}-{safe_channel}-transcript.txt",
        )
        date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        time_str = f"{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        transcript_header = f"{date_str} {time_str} — {guild_name} — {channel.name}"
        file_content = transcript_header + "\n\n"
        if recap:
            file_content += recap + "\n\n"
        file_content += transcript_plain
        try:
            with open(transcript_saved_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            logger.debug("Saved transcript to %s", transcript_saved_path)
        except OSError as e:
            logger.warning(
                "Could not save transcript to %s: %s", transcript_saved_path, e
            )

        try:
            recording_paths = []
            for temp_path, user_id in temp_files:
                file_name = f"{timestamp}-{safe_guild}-{safe_channel}-user{user_id}.wav"
                dest = os.path.join(_watson_recordings_dir, file_name)
                try:
                    shutil.copy2(temp_path, dest)
                    recording_paths.append(dest)
                except OSError as e:
                    logger.warning("Could not copy %s to %s: %s", temp_path, dest, e)
            lines = [f"- `{p}`" for p in recording_paths]
            if os.path.exists(transcript_saved_path):
                lines.append(f"- `{transcript_saved_path}` (transcript)")
            if lines:
                await status_msg.edit(
                    content="✅ **Done.**\n\n"
                    + recap_block
                    + "📁 Saved to recordings:\n"
                    + "\n".join(lines)
                )
            else:
                await status_msg.edit(
                    content="✅ **Done.**\n\n" + recap_block + "(no files saved)"
                )
        except discord.DiscordException as e:
            logger.exception(
                "Failed to send message to channel (guild %s): %s", guild_id, e
            )
            try:
                await channel.send(
                    "⚠️ Processing finished but failed to post. Check bot permissions and logs."
                )
            except discord.DiscordException:
                pass

    finally:
        transcribing_guilds.discard(guild_id)
        logger.debug("Removed guild %s from transcribing_guilds", guild_id)
        if os.path.isdir(temp_guild_dir):
            def _rmtree() -> None:
                try:
                    shutil.rmtree(temp_guild_dir)
                except OSError as e:
                    logger.warning("Could not remove temp dir %s: %s", temp_guild_dir, e)
            try:
                await asyncio.to_thread(_rmtree)
            except Exception:
                _rmtree()
        del sink
        await asyncio.to_thread(gc.collect)
        _log_memory("transcription_done")
        logger.info(
            "Session finished for guild %s (%s), saved %d recording(s)",
            guild_id,
            guild_name,
            len(temp_files),
        )


SHUTDOWN_WAIT_TRANSCRIPTION_SEC = 300


def _create_bot() -> commands.Bot:
    """Create and configure the bot. Must be called when the event loop is already running."""
    b = commands.Bot(command_prefix=_bot_prefix, intents=intents)
    b.add_listener(on_ready)
    b.add_listener(on_voice_state_update)
    b.command(name="check")(check)
    b.command(name="join")(join)
    b.command(name="record")(record)
    b.command(name="stop")(stop)
    b.command(name="leave")(leave)
    return b


async def _run_bot_with_graceful_shutdown(token: str) -> None:
    """
    Run the bot until SIGTERM/SIGINT; then wait for active transcriptions
    (up to SHUTDOWN_WAIT_TRANSCRIPTION_SEC), close the Discord connection, and exit.
    """
    global bot
    bot = _create_bot()

    shutdown_ev = asyncio.Event()

    def _on_signal() -> None:
        shutdown_ev.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, _on_signal
            )
    except NotImplementedError:
        signal.signal(signal.SIGTERM, lambda s, f: _on_signal())
        signal.signal(signal.SIGINT, lambda s, f: _on_signal())

    bot_task = asyncio.create_task(bot.start(token))
    await shutdown_ev.wait()
    logger.info("Shutdown requested, waiting for active transcriptions...")
    deadline = time.monotonic() + SHUTDOWN_WAIT_TRANSCRIPTION_SEC
    while transcribing_guilds and time.monotonic() < deadline:
        await asyncio.sleep(1)
    if transcribing_guilds:
        logger.warning(
            "Shutdown timeout, %d guild(s) still transcribing",
            len(transcribing_guilds),
        )
    await bot.close()
    try:
        await asyncio.wait_for(bot_task, timeout=10.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    logger.info("Bot stopped.")


def _main() -> None:
    """Entry point: check token, then run bot with graceful shutdown."""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error("DISCORD_TOKEN not found in .env")
        sys.exit("Set DISCORD_TOKEN in .env (see .env.example)")
    logger.info("Starting bot...")
    try:
        asyncio.run(_run_bot_with_graceful_shutdown(token))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    _main()
