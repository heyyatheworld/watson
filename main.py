"""Discord bot that records voice channel audio and transcribes it with Whisper."""

import asyncio
import logging
import os

import discord
import whisper
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

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

# Load Opus before creating the bot (required on macOS Homebrew)
try:
    discord.opus.load_opus('/opt/homebrew/lib/libopus.dylib')
    logger.info("Opus loaded")
except Exception as e:
    logger.info("Opus fallback: %s", e)

logger.info("Loading Whisper model (turbo)...")
model = whisper.load_model("turbo")
logger.info("Whisper ready")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    """Log bot name, ID, and guild count when the bot comes online."""
    logger.info("Watson online — %s (ID: %s), guilds: %d", bot.user.name, bot.user.id, len(bot.guilds))


@bot.event
async def on_voice_state_update(member, before, after):
    """Stop recording when the last human leaves the voice channel."""
    if before.channel is not None:
        voice_client = member.guild.voice_client
        if voice_client and voice_client.channel == before.channel:
            human_members = [m for m in before.channel.members if not m.bot]
            if len(human_members) == 0:
                if voice_client.recording:
                    logger.info("Channel %s empty, ending session", before.channel.name)
                    voice_client.stop_recording()
                    await asyncio.sleep(1)


@bot.command()
async def check(ctx):
    """Reply with connection status and bot permissions in the current channel."""
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
    if ctx.voice_client:
        return await ctx.send("I'm already in a channel! 🎙")
    if ctx.author.voice:
        await ctx.author.voice.channel.connect()
        await ctx.send("🎩 Joined. Ready.")
    else:
        await ctx.send("Join a voice channel first.")


@bot.command()
async def record(ctx):
    """Start recording and transcribing voice in the current channel."""
    voice = ctx.voice_client
    if not voice:
        return await ctx.send("Invite me with !join first.")
    if voice.recording:
        return await ctx.send("⚠️ Recording is already in progress.")

    await ctx.send("⏺ **Recording started.**")
    voice.start_recording(discord.sinks.WaveSink(), once_done, ctx.channel)


@bot.command()
async def stop(ctx):
    """Stop the current recording and process the transcript."""
    voice = ctx.voice_client
    if voice and voice.recording:
        voice.stop_recording()
        await ctx.send("⏹ Recording stopped. Building transcript...")
    else:
        await ctx.send("I'm not recording right now.")


@bot.command()
async def leave(ctx):
    """Leave the current voice channel."""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Bye!")
    else:
        await ctx.send("I'm not in a voice channel.")


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    """Process recorded audio: transcribe with Whisper and post transcript to the channel."""
    if not sink.audio_data:
        await channel.send("📭 Recording is empty (silence).")
        return

    guild_id = channel.guild.id
    status_msg = await channel.send("⚙️ **Watson is processing audio...**")

    all_phrases = []
    junk_phrases = ["editor", "subtitles", "thanks for watching", "to be continued", "а.семкин", "субтитры", "продолжение следует", "спасибо за просмотр"]
    temp_files = []

    try:
        for user_id, audio in sink.audio_data.items():
            file_name = f"temp_{guild_id}_{user_id}.wav"
            temp_files.append(file_name)

            audio.file.seek(0)
            data = audio.file.read()

            if len(data) < 2000:
                continue

            with open(file_name, "wb") as f:
                f.write(data)

            try:
                result = await asyncio.to_thread(
                    model.transcribe,
                    file_name,
                    language="russian",
                    fp16=False
                )

                user_obj = bot.get_user(user_id)
                username = user_obj.display_name if user_obj else f"User {user_id}"

                for segment in result['segments']:
                    text = segment['text'].strip()
                    if not any(junk in text.lower() for junk in junk_phrases) and len(text) > 1:
                        all_phrases.append({
                            'time': segment['start'],
                            'user': username,
                            'text': text
                        })
            except Exception as e:
                logger.exception("Whisper error for user %s: %s", user_id, e)

        all_phrases.sort(key=lambda x: x['time'])

        raw_transcript = ""
        for p in all_phrases:
            m, s = divmod(int(p['time']), 60)
            raw_transcript += f"[{m:02d}:{s:02d}] **{p['user']}**: {p['text']}\n"

        if not raw_transcript:
            await status_msg.edit(content="😶 Could not recognize any speech.")
            return

        header = f"📋 **TRANSCRIPT ({channel.guild.name})**\n\n"

        if len(header + raw_transcript) > 2000:
            file_path = f"transcript_{guild_id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(raw_transcript.replace("**", ""))

            await channel.send(header + "See attachment:", file=discord.File(file_path))
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            await status_msg.edit(content=header + raw_transcript)

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Session finished for guild %s", guild_id)


token = os.getenv("DISCORD_TOKEN")
if not token:
    logger.error("DISCORD_TOKEN not found in .env")
    raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example)")
bot.run(token)
