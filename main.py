"""Discord bot that records voice channel audio and transcribes it with faster-whisper."""

import asyncio
import gc
import logging
import os
import shutil
import tempfile
from datetime import datetime

import psutil
import discord
from discord.ext import commands
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# Temporary directory for WAV recordings and transcript files (avoids cluttering project dir)
_watson_temp_dir = os.getenv("WATSON_TEMP_DIR")
if not _watson_temp_dir:
    _watson_temp_dir = os.path.join(tempfile.gettempdir(), "watson")
os.makedirs(_watson_temp_dir, exist_ok=True)

# Persistent directory for saved voice recordings
_watson_recordings_dir = os.getenv("WATSON_RECORDINGS_DIR")
if not _watson_recordings_dir:
    _watson_recordings_dir = os.path.join(_watson_temp_dir, "recordings")
os.makedirs(_watson_recordings_dir, exist_ok=True)

# Logging: level from env, optional file output
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

logging.getLogger('discord.voicereader').setLevel(logging.ERROR)
logging.getLogger('discord.voicereader').propagate = False

# Load Opus before creating the bot (required on macOS Homebrew)
try:
    discord.opus.load_opus('/opt/homebrew/lib/libopus.dylib')
    logger.info("Opus loaded")
except Exception as e:
    logger.info("Opus fallback: %s", e)
_log_memory("after_opus")

logger.info("Loading Whisper model (turbo)...")
model = WhisperModel("turbo", device="cpu", compute_type="int8")
logger.info("Whisper ready")
logger.info("Temp dir for recordings: %s", _watson_temp_dir)
_log_memory("after_whisper_load")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Guilds currently running Whisper transcription; no new recording until done
transcribing_guilds = set()

# Max recording length (minutes); after this, recording stops and a new one can be started
MAX_RECORDING_MINUTES = int(os.getenv("RECORDING_MAX_MINUTES", "30"))
MAX_RECORDING_SECONDS = MAX_RECORDING_MINUTES * 60

# Whisper transcription: language (ISO code) and beam_size
# None = auto-detect (faster-whisper does not accept empty string)
_transcript_lang = (os.getenv("TRANSCRIPT_LANGUAGE") or "").strip()
TRANSCRIPT_LANGUAGE = _transcript_lang or None
TRANSCRIPT_BEAM_SIZE = int(os.getenv("TRANSCRIPT_BEAM_SIZE", "5"))


@bot.event
async def on_ready():
    """Log bot name, ID, and guild count when the bot comes online."""
    logger.info("Watson online — %s (ID: %s), guilds: %d", bot.user.name, bot.user.id, len(bot.guilds))
    _log_memory("on_ready")
    for guild in bot.guilds:
        logger.info("  Guild: %s (ID: %s)", guild.name, guild.id)


@bot.event
async def on_voice_state_update(member, before, after):
    """When the last human leaves the bot's voice channel: stop recording (triggers transcription) and leave."""
    if before.channel is None:
        return
    voice_client = member.guild.voice_client
    if not voice_client or voice_client.channel != before.channel:
        return
    # After this member left: who remains in the channel (before.channel.members may still include member in some versions)
    humans_remaining = [m for m in before.channel.members if m != member and not m.bot]
    logger.debug("Voice state: %s left %s (guild %s), humans remaining: %d", member.display_name, before.channel.name, member.guild.id, len(humans_remaining))
    if len(humans_remaining) != 0:
        return
    # Bot is alone (or channel empty): stop recording then leave
    if voice_client.recording:
        logger.info("Channel %s (guild %s) empty, stopping recording → transcription will run", before.channel.name, member.guild.id)
        voice_client.stop_recording()
    await voice_client.disconnect()
    logger.info("Left voice channel %s (guild %s), no users left", before.channel.name, member.guild.id)


@bot.command()
async def check(ctx):
    """Reply with connection status and bot permissions in the current channel."""
    logger.info("!check from %s in %s/#%s (guild %s)", ctx.author, ctx.guild.name, ctx.channel.name, ctx.guild.id)
    perms = ctx.channel.permissions_for(ctx.me)
    status = [
        f"✅ **Connection:** OK",
        f"🎤 **Voice channel:** {'✅' if ctx.author.voice else '❌ (you are not in a channel)'}",
        f"📝 **Send messages:** {'✅' if perms.send_messages else '❌'}",
        f"📎 **Attach files:** {'✅' if perms.attach_files else '❌'}",
        f"📜 **Read history:** {'✅' if perms.read_message_history else '❌'}",
        f"🎙 **Speak:** {'✅' if perms.speak else '❌'}"
    ]
    embed = discord.Embed(
        title="Watson system check",
        description="\n".join(status),
        color=discord.Color.blue() if perms.attach_files else discord.Color.red()
    )
    await ctx.send(embed=embed)


@bot.command()
async def join(ctx):
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


async def _enforce_recording_limit(guild_id: int, channel_id: int):
    """After MAX_RECORDING_SECONDS, stop the current recording and notify; user can start a new one with !record."""
    await asyncio.sleep(MAX_RECORDING_SECONDS)
    guild = bot.get_guild(guild_id)
    if not guild:
        return
    voice = guild.voice_client
    if voice and voice.recording:
        logger.info("Recording limit (%d min) reached for guild %s, stopping", MAX_RECORDING_MINUTES, guild_id)
        voice.stop_recording()
        ch = guild.get_channel(channel_id)
        if ch:
            try:
                await ch.send(f"⏱ **{MAX_RECORDING_MINUTES} min limit reached.** Use `!record` to start a new recording.")
            except discord.DiscordException:
                pass


@bot.command()
async def record(ctx):
    """Start recording and transcribing voice in the current channel (max length set by RECORDING_MAX_MINUTES, default 30 min)."""
    logger.info("!record from %s in guild %s (channel %s)", ctx.author, ctx.guild.id, ctx.channel.name)
    voice = ctx.voice_client
    if not voice:
        logger.debug("Rejected: bot not in voice channel")
        return await ctx.send("Invite me with !join first.")
    if voice.recording:
        logger.debug("Rejected: recording already in progress")
        return await ctx.send("⚠️ Recording is already in progress.")
    if ctx.guild.id in transcribing_guilds:
        logger.debug("Rejected: transcription in progress for guild %s", ctx.guild.id)
        return await ctx.send("⚠️ Previous recording is still being transcribed. Wait for it to finish.")

    logger.info("Recording started in %s (guild %s), limit %d min", voice.channel.name, ctx.guild.id, MAX_RECORDING_MINUTES)
    await ctx.send(f"⏺ **Recording started.** (max {MAX_RECORDING_MINUTES} min)")
    voice.start_recording(discord.sinks.WaveSink(), once_done, ctx.channel)
    asyncio.create_task(_enforce_recording_limit(ctx.guild.id, ctx.channel.id))


@bot.command()
async def stop(ctx):
    """Stop the current recording and process the transcript."""
    logger.info("!stop from %s in guild %s", ctx.author, ctx.guild.id)
    voice = ctx.voice_client
    if voice and voice.recording:
        logger.info("Stopping recording in %s (guild %s), transcript will follow", voice.channel.name, ctx.guild.id)
        voice.stop_recording()
        await ctx.send("⏹ Recording stopped. Building transcript...")
    else:
        logger.debug("Rejected: not recording")
        await ctx.send("I'm not recording right now.")


@bot.command()
async def leave(ctx):
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


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    """Process recorded audio: transcribe with Whisper and post transcript to the channel."""
    guild_id = channel.guild.id
    guild_name = channel.guild.name
    num_participants = len(sink.audio_data) if sink.audio_data else 0

    logger.info("once_done: guild %s (%s), channel %s, participants: %d", guild_id, guild_name, channel.name, num_participants)

    if not sink.audio_data:
        logger.info("Empty recording for guild %s, skipping transcription", guild_id)
        await channel.send("📭 Recording is empty (silence).")
        return

    transcribing_guilds.add(guild_id)
    logger.debug("Added guild %s to transcribing_guilds", guild_id)
    _log_memory("transcription_start")
    status_msg = await channel.send("⚙️ **Watson is processing audio...**")

    all_phrases = []
    junk_phrases = ["editor", "subtitles", "thanks for watching", "to be continued", "а.семкин", "субтитры", "продолжение следует", "спасибо за просмотр"]
    # (temp_path, user_id) for cleanup and later copy to permanent storage
    temp_files = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_guild = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in guild_name)
    safe_channel = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in channel.name)
    temp_guild_dir = os.path.join(_watson_temp_dir, str(guild_id))
    os.makedirs(temp_guild_dir, exist_ok=True)

    try:
        for user_id, audio in sink.audio_data.items():
            temp_path = os.path.join(temp_guild_dir, f"temp_{user_id}.wav")

            audio.file.seek(0)
            data = audio.file.read()
            data_len = len(data)

            if data_len < 2000:
                logger.debug("Skipping user %s: audio too short (%d bytes)", user_id, data_len)
                continue

            with open(temp_path, "wb") as f:
                f.write(data)
            logger.debug("Saved to temp %s (%d bytes), user %s", temp_path, data_len, user_id)
            temp_files.append((temp_path, user_id))

            try:
                def _transcribe(path: str):
                    segments_iter, _ = model.transcribe(path, beam_size=TRANSCRIPT_BEAM_SIZE, language=TRANSCRIPT_LANGUAGE)
                    return list(segments_iter)

                segments_list = await asyncio.to_thread(_transcribe, temp_path)
                num_segments = len(segments_list)
                logger.info("Transcribed user %s: %d segments", user_id, num_segments)
                _log_memory("after_transcribe_user_%s" % user_id)

                user_obj = bot.get_user(user_id)
                username = user_obj.display_name if user_obj else f"User {user_id}"

                for seg in segments_list:
                    text = (seg.text or "").strip()
                    if not any(junk in text.lower() for junk in junk_phrases) and len(text) > 1:
                        all_phrases.append({
                            'time': seg.start,
                            'user': username,
                            'text': text
                        })
                del segments_list
            except Exception as e:
                logger.exception("Whisper error for user %s: %s", user_id, e)

        all_phrases.sort(key=lambda x: x['time'])
        logger.debug("Collected %d phrases", len(all_phrases))

        raw_transcript = build_transcript_lines(all_phrases)

        if not raw_transcript:
            logger.info("No speech recognized for guild %s", guild_id)
            await status_msg.edit(content="😶 Could not recognize any speech.")
            return

        header = f"📋 **TRANSCRIPT ({channel.guild.name})**\n\n"
        total_len = len(header) + len(raw_transcript)
        transcript_plain = raw_transcript.replace("**", "")

        # Save transcript alongside voice recordings (same name prefix for sorting)
        transcript_saved_path = os.path.join(
            _watson_recordings_dir,
            f"{timestamp}-{safe_guild}-{safe_channel}-transcript.txt",
        )
        try:
            with open(transcript_saved_path, "w", encoding="utf-8") as f:
                f.write(transcript_plain)
            logger.debug("Saved transcript to %s", transcript_saved_path)
        except OSError as e:
            logger.warning("Could not save transcript to %s: %s", transcript_saved_path, e)

        try:
            if total_len > 100:
                file_path = os.path.join(temp_guild_dir, "transcript.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(transcript_plain)
                logger.info("Transcript too long (%d chars), sending as file %s (guild %s)", total_len, file_path, guild_id)
                await channel.send(header + "See attachment:", file=discord.File(file_path))
            else:
                logger.info("Sending transcript (%d chars) to channel (guild %s)", total_len, guild_id)
                await status_msg.edit(content=header + raw_transcript)

            # Copy temp WAVs to permanent storage, then we clear temp in finally
            recording_paths = []
            for temp_path, user_id in temp_files:
                file_name = f"{timestamp}-{safe_guild}-{safe_channel}-user{user_id}.wav"
                dest = os.path.join(_watson_recordings_dir, file_name)
                try:
                    shutil.copy2(temp_path, dest)
                    recording_paths.append(dest)
                except OSError as e:
                    logger.warning("Could not copy %s to %s: %s", temp_path, dest, e)
            if recording_paths or transcript_saved_path:
                lines = [f"- `{p}`" for p in recording_paths]
                if os.path.exists(transcript_saved_path):
                    lines.append(f"- `{transcript_saved_path}` (transcript)")
                if lines:
                    await channel.send(f"📁 Saved to recordings:\n" + "\n".join(lines))
        except discord.DiscordException as e:
            logger.exception("Failed to send transcript to channel (guild %s): %s", guild_id, e)
            try:
                await channel.send("⚠️ Transcript ready but failed to post. Check bot permissions and logs.")
            except discord.DiscordException:
                pass

    finally:
        transcribing_guilds.discard(guild_id)
        logger.debug("Removed guild %s from transcribing_guilds", guild_id)
        # Remove this guild's temp subdir (WAVs + transcript)
        if os.path.isdir(temp_guild_dir):
            try:
                shutil.rmtree(temp_guild_dir)
            except OSError as e:
                logger.warning("Could not remove temp dir %s: %s", temp_guild_dir, e)
        del sink
        gc.collect()
        _log_memory("transcription_done")
        logger.info("Session finished for guild %s (%s), saved %d recording(s)", guild_id, guild_name, len(temp_files))


token = os.getenv("DISCORD_TOKEN")
if not token:
    logger.error("DISCORD_TOKEN not found in .env")
    raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example)")
logger.info("Starting bot...")
bot.run(token)
