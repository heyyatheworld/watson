"""Discord bot that records voice channel audio and transcribes it with Whisper."""

import os
import time

import discord
import requests
from dotenv import load_dotenv
load_dotenv()

from discord.ext import commands
import whisper

# Load Opus before creating the bot (required on macOS Homebrew)
try:
    discord.opus.load_opus('/opt/homebrew/lib/libopus.dylib')
    print("✅ Opus loaded")
except Exception as e:
    print(f"❌ Opus load error: {e}")

print("Loading Whisper model...")
model = whisper.load_model("turbo")
print("Whisper ready.")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    """Log bot name, ID, and connected guilds when the bot comes online."""
    print(f'--- Watson Online ---')
    print(f'Bot: {bot.user.name} (ID: {bot.user.id})')
    print('Guilds:')
    for guild in bot.guilds:
        print(f'  - {guild.name} (ID: {guild.id})')
    print('----------------------')

@bot.command()
async def check(ctx):
    """Reply with connection status and bot permissions in the current channel."""
    permissions = ctx.channel.permissions_for(ctx.me)
    status_message = (
        "✅ **Connection OK**\n"
        f"I see you, {ctx.author.mention}.\n\n"
        "**Permissions in this channel:**\n"
        f"- Send messages: {'✅' if permissions.send_messages else '❌'}\n"
        f"- Read history: {'✅' if permissions.read_message_history else '❌'}\n"
        f"- Administrator: {'👑 Yes' if permissions.administrator else 'No'}"
    )
    await ctx.send(status_message)

@bot.command()
async def join(ctx):
    """Join the voice channel the author is in."""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"🎩 Joined `{channel.name}`. Ready to listen.")
    else:
        await ctx.send("You need to be in a voice channel first.")

@bot.command()
async def leave(ctx):
    """Leave the current voice channel."""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left the channel.")
    else:
        await ctx.send("I'm not in a voice channel.")

connections = {}
start_times = {}

@bot.command()
async def record(ctx):
    """Start recording and transcribing voice in the current channel."""
    voice = ctx.voice_client
    if not voice:
        await ctx.send("Join first with !join")
        return

    start_times[ctx.guild.id] = time.time()
    await ctx.send("⏺ **Recording and transcription started.** Speak in turn...")
    voice.start_recording(discord.sinks.WaveSink(), once_done, ctx.channel)


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    """Process recorded audio: transcribe with Whisper, post transcript, then run Ollama analysis."""
    if not sink.audio_data:
        await channel.send("📭 Recording is empty. We may have been silent.")
        return

    await channel.send("⚙️ **Watson is processing audio with the Turbo model...**")

    all_phrases = []
    junk_phrases = [
        "Редактор", "Корректор", "Субтитры", "Продолжение следует", "Спасибо за просмотр", "А.Семкин",
        "Editor", "Subtitles", "To be continued", "Thanks for watching"
    ]

    for user_id, audio in sink.audio_data.items():
        file_name = f"temp_{user_id}.wav"
        audio.file.seek(0)
        data = audio.file.read()

        if len(data) < 2000:
            continue

        with open(file_name, "wb") as f:
            f.write(data)

        try:
            result = model.transcribe(
                file_name,
                language="russian",
                fp16=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )

            user_obj = bot.get_user(user_id)
            username = user_obj.display_name if user_obj else f"User {user_id}"

            for segment in result['segments']:
                text = segment['text'].strip()
                is_junk = any(junk.lower() in text.lower() for junk in junk_phrases)
                if len(text) > 1 and not is_junk:
                    all_phrases.append({
                        'time': segment['start'],
                        'user': username,
                        'text': text
                    })

            os.remove(file_name)

        except Exception as e:
            print(f"❌ Whisper error for {user_id}: {e}")

    all_phrases.sort(key=lambda x: x['time'])

    raw_transcript_lines = []
    for p in all_phrases:
        m, s = divmod(int(p['time']), 60)
        timestamp = f"[{m:02d}:{s:02d}]"
        raw_transcript_lines.append(f"{timestamp} **{p['user']}**: {p['text']}")

    raw_text = "\n".join(raw_transcript_lines)

    if not raw_text:
        await channel.send("😶 Watson could not make out any words.")
        return

    report_header = "📋 **TRANSCRIPT**\n\n"
    if len(report_header + raw_text) > 2000:
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(raw_text.replace("**", ""))
        await channel.send(report_header + "Text too long, attaching file:", file=discord.File("transcript.txt"))
    else:
        await channel.send(report_header + raw_text)

    await channel.send("🧠 **Watson is analyzing the conversation...**")

    prompt = f"""
You are the Watson AI assistant. You are given a transcript of a Discord voice conversation.
Your tasks:
1. Fix obvious recognition errors (e.g. 'pycord' -> Pycord, 'watson' -> Watson).
2. Write a brief meeting summary (what was discussed).
3. List key action items if any were mentioned.

Write in English, concisely.

Transcript:
{raw_text}
"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            ai_analysis = response.json().get('response', '')
            analysis_msg = "📝 **ANALYSIS & CONCLUSIONS:**\n\n" + ai_analysis
            if len(analysis_msg) > 2000:
                with open("analysis.txt", "w", encoding="utf-8") as f:
                    f.write(ai_analysis)
                await channel.send("📝 Analysis ready (see attachment):", file=discord.File("analysis.txt"))
            else:
                await channel.send(analysis_msg)
        else:
            await channel.send("⚠️ Ollama returned an error. Check if the server is running.")

    except Exception as e:
        print(f"Ollama error: {e}")
        await channel.send("⚠️ Could not connect to Ollama for analysis.")


@bot.command()
async def stop(ctx):
    """Stop the current recording."""
    voice = ctx.voice_client
    if voice and voice.recording:
        try:
            voice.stop_recording()
            if ctx.guild.id in connections:
                del connections[ctx.guild.id]
        except Exception as e:
            await ctx.send(f"⚠️ Stop error: {e}")
    else:
        await ctx.send("I'm not recording right now.")

token = os.getenv("DISCORD_TOKEN")
if not token:
    raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example)")
bot.run(token)
