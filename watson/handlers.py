"""Discord listeners, commands, and recording pipeline."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import shutil
from datetime import datetime, timezone

import discord
from discord.ext import commands

from watson.config import SETTINGS
from watson.paths import format_saved_paths_for_discord
from watson.permissions import ensure_voice_operator
from watson.recap import get_recap_sync
from watson import state
from watson.transcribe import build_transcript_lines, get_whisper_model

logger = logging.getLogger(__name__)

_whisper_jobs_semaphore: asyncio.Semaphore | None = None


def _whisper_jobs_slot_limiter() -> asyncio.Semaphore:
    """Bound concurrent Whisper pipelines across all guilds (lazy singleton)."""
    global _whisper_jobs_semaphore
    if _whisper_jobs_semaphore is None:
        n = SETTINGS.max_concurrent_transcriptions
        _whisper_jobs_semaphore = asyncio.Semaphore(n)
        logger.info(
            "Transcription concurrency limit: %d (WATSON_MAX_CONCURRENT_TRANSCRIPTIONS)",
            n,
        )
    return _whisper_jobs_semaphore


intents = discord.Intents.default()
intents.message_content = True
intents.members = True


async def on_ready() -> None:
    """Log bot name, ID, and guild count when the bot comes online."""
    assert state.bot is not None
    logger.info(
        "Watson online — %s (ID: %s), guilds: %d",
        state.bot.user.name,
        state.bot.user.id,
        len(state.bot.guilds),
    )
    for guild in state.bot.guilds:
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
    if not await ensure_voice_operator(ctx):
        return
    logger.info("!join from %s in guild %s", ctx.author, ctx.guild.id)
    if ctx.voice_client:
        logger.debug("Rejected: already in channel %s", ctx.voice_client.channel.name)
        return await ctx.send("I'm already in a channel! 🎙")
    if ctx.author.voice:
        ch = ctx.author.voice.channel
        logger.info(
            "Attempting voice connect: guild=%s, channel=%s (id=%s, bitrate=%s, user_limit=%s)",
            ctx.guild.id,
            ch.name,
            ch.id,
            getattr(ch, "bitrate", None),
            getattr(ch, "user_limit", None),
        )
        try:
            connect_task = asyncio.create_task(ch.connect(timeout=90.0, reconnect=True))
            while not connect_task.done():
                await asyncio.sleep(0.25)
                if ctx.guild.voice_client is None:
                    connect_task.cancel()
                    try:
                        await connect_task
                    except asyncio.CancelledError:
                        pass
                    return await ctx.send("Left before connection finished.")
            await connect_task
            vc = ctx.guild.voice_client
            logger.info(
                "Joined voice channel %s (id=%s) in guild %s, voice_client=%r, ws=%r",
                ch.name,
                ch.id,
                ctx.guild.id,
                vc,
                getattr(vc, "ws", None) if vc else None,
            )
            await ctx.send("🎩 Joined. Ready.")
        except asyncio.CancelledError:
            pass
        except discord.errors.ConnectionClosed as e:
            logger.exception(
                "Voice websocket closed during connect (guild=%s, channel=%s, code=%s): %s",
                ctx.guild.id,
                ch.id,
                getattr(e, "code", None),
                e,
            )
            if getattr(e, "code", None) == 4017:
                logger.warning(
                    "Voice close code 4017: DAVE (Discord mandatory E2EE for voice) validation "
                    "failed — Discord rejected the handshake; upgrading Py-cord to a "
                    "DAVE-compatible release is required once available (not fixable inside Watson)."
                )
            await ctx.send(
                "⚠️ Could not connect to voice (ConnectionClosed from Discord). "
                "Check bot logs for details."
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during voice connect (guild=%s, channel=%s): %s",
                ctx.guild.id,
                ch.id,
                e,
            )
            await ctx.send(
                "⚠️ Could not connect to voice (unexpected error). "
                "Check bot logs for details."
            )
    else:
        logger.debug("Rejected: author not in voice channel")
        await ctx.send("Join a voice channel first.")


async def _enforce_recording_limit(guild_id: int, channel_id: int) -> None:
    """
    After max recording duration, stop the current recording and notify.
    Sends a warning shortly before the limit.
    """
    assert state.bot is not None
    max_sec = SETTINGS.max_recording_seconds
    warn_sec = SETTINGS.warning_before_stop_seconds
    warning_after = max(0, max_sec - warn_sec)
    await asyncio.sleep(warning_after)
    guild = state.bot.get_guild(guild_id)
    if not guild:
        return
    ch = guild.get_channel(channel_id)
    voice = guild.voice_client
    if voice and voice.recording and ch and warn_sec < max_sec:
        try:
            await ch.send(
                f"⚠️ **{SETTINGS.warning_before_stop_minutes} min left** until auto-stop "
                f"(limit {SETTINGS.max_recording_minutes} min). "
                "Stop manually with `!stop`."
            )
        except discord.DiscordException:
            pass
    remaining = max_sec - warning_after
    await asyncio.sleep(remaining)
    guild = state.bot.get_guild(guild_id)
    if not guild:
        return
    voice = guild.voice_client
    if voice and voice.recording:
        logger.info(
            "Recording limit (%d min) reached for guild %s, stopping",
            SETTINGS.max_recording_minutes,
            guild_id,
        )
        voice.stop_recording()
        ch = guild.get_channel(channel_id)
        if ch:
            try:
                await ch.send(
                    f"⏱ **{SETTINGS.max_recording_minutes} min limit reached.** "
                    "Use `!record` to start a new recording."
                )
            except discord.DiscordException:
                pass


async def record(ctx) -> None:
    """Start recording in the current voice channel."""
    if not await ensure_voice_operator(ctx):
        return
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
    if ctx.guild.id in state.transcribing_guilds:
        logger.debug("Rejected: transcription in progress for guild %s", ctx.guild.id)
        return await ctx.send(
            "⚠️ Previous recording is still being transcribed. Wait for it to finish."
        )

    logger.info(
        "Recording started in %s (guild %s), limit %d min",
        voice.channel.name,
        ctx.guild.id,
        SETTINGS.max_recording_minutes,
    )
    await ctx.send(f"⏺ **Recording started.** (max {SETTINGS.max_recording_minutes} min)")
    voice.start_recording(discord.sinks.WaveSink(), once_done, ctx.channel)
    asyncio.create_task(_enforce_recording_limit(ctx.guild.id, ctx.channel.id))


async def stop(ctx) -> None:
    """Stop the current recording and run transcription and recap."""
    if not await ensure_voice_operator(ctx):
        return
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
    if not await ensure_voice_operator(ctx):
        return
    logger.info("!leave from %s in guild %s", ctx.author, ctx.guild.id)
    if ctx.voice_client:
        ch_name = ctx.voice_client.channel.name
        await ctx.voice_client.disconnect()
        logger.info("Left voice channel %s (guild %s)", ch_name, ctx.guild.id)
        await ctx.send("Bye!")
    else:
        logger.debug("Rejected: not in voice channel")
        await ctx.send("I'm not in a voice channel.")


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args) -> None:
    """
    Process recorded audio: transcribe with Whisper, save WAV and transcript to recordings,
    post recap (if Ollama enabled) and file links to the channel.
    """
    assert state.bot is not None
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

    state.transcribing_guilds.add(guild_id)
    logger.debug("Added guild %s to transcribing_guilds", guild_id)
    status_msg = await channel.send("⚙️ **Watson is processing audio...**")

    async def _run_transcription_pipeline() -> None:
            all_phrases = []
            junk_phrases = SETTINGS.transcript_junk_phrases
            temp_files = []
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_guild = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in guild_name)
            safe_channel = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in channel.name)
            temp_guild_dir = os.path.join(SETTINGS.temp_dir, str(guild_id))
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
                            whisper = get_whisper_model()
                            segments_iter, _ = whisper.transcribe(
                                path,
                                beam_size=SETTINGS.transcript_beam_size,
                                language=SETTINGS.transcript_language,
                            )
                            return list(segments_iter)

                        segments_list = await asyncio.to_thread(_transcribe, temp_path)
                        await asyncio.sleep(0)
                        num_segments = len(segments_list)
                        logger.info("Transcribed user %s: %d segments", user_id, num_segments)

                        user_obj = state.bot.get_user(user_id)
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
                if SETTINGS.ollama_recap_model:
                    recap = await asyncio.to_thread(get_recap_sync, transcript_plain, SETTINGS, logger)
                    await asyncio.sleep(0)
                recap_block = (recap + "\n\n") if recap else ""

                transcript_saved_path = os.path.join(
                    SETTINGS.recordings_dir,
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
                        dest = os.path.join(SETTINGS.recordings_dir, file_name)
                        try:
                            shutil.copy2(temp_path, dest)
                            recording_paths.append(dest)
                        except OSError as e:
                            logger.warning("Could not copy %s to %s: %s", temp_path, dest, e)
                    paths_for_message = list(recording_paths)
                    kinds = ["wav"] * len(recording_paths)
                    if os.path.exists(transcript_saved_path):
                        paths_for_message.append(transcript_saved_path)
                        kinds.append("transcript")
                    display_paths = format_saved_paths_for_discord(
                        paths_for_message,
                        SETTINGS.recordings_dir,
                        SETTINGS.discord_hide_paths,
                    )
                    lines = []
                    for p, kind in zip(display_paths, kinds):
                        suffix = " (transcript)" if kind == "transcript" else ""
                        lines.append(f"- `{p}`{suffix}")
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
                state.transcribing_guilds.discard(guild_id)
                logger.debug("Removed guild %s from transcribing_guilds", guild_id)
                if os.path.isdir(temp_guild_dir):
                    try:
                        await asyncio.to_thread(shutil.rmtree, temp_guild_dir)
                    except OSError as e:
                        logger.warning("Could not remove temp dir %s: %s", temp_guild_dir, e)
                del sink
                await asyncio.to_thread(gc.collect)
                logger.info(
                    "Session finished for guild %s (%s), saved %d recording(s)",
                    guild_id,
                    guild_name,
                    len(temp_files),
                )


    async with _whisper_jobs_slot_limiter():
        await _run_transcription_pipeline()

def create_bot() -> commands.Bot:
    """Create and configure the bot."""
    b = commands.Bot(command_prefix=SETTINGS.bot_prefix, intents=intents)
    b.add_listener(on_ready)
    b.add_listener(on_voice_state_update)
    b.command(name="check")(check)
    b.command(name="join")(join)
    b.command(name="record")(record)
    b.command(name="stop")(stop)
    b.command(name="leave")(leave)
    state.bot = b
    return b
