"""Discord bot that records voice channel audio and transcribes it with Whisper."""

import os
import time
import httpx
import asyncio
import discord
import requests
from discord.ext import commands
import whisper

from dotenv import load_dotenv
load_dotenv()

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

connections = {}
start_times = {}

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

async def ask_ollama(prompt):
    """Асинхронный запрос к Ollama через httpx."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3:latest",
        "prompt": prompt,
        "stream": False
    }
    # Используем бесконечный таймаут, так как анализ длинного текста может занять время
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('response', '')
            return f"⚠️ Ошибка Ollama: {response.status_code}"
        except Exception as e:
            return f"⚠️ Ошибка подключения к Ollama: {e}"

async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    """Безопасная обработка аудио для нескольких серверов одновременно."""
    if not sink.audio_data:
        await channel.send("📭 Запись пуста или была тишина.")
        return

    guild_id = channel.guild.id
    guild_name = channel.guild.name
    status_msg = await channel.send(f"⚙️ **Ватсон (сервер: {guild_name}) обрабатывает аудио...**")
    
    print(f"🚀 Начата обработка для сервера: {guild_name} (ID: {guild_id})")

    all_phrases = []
    junk_phrases = ["Редактор", "Субтитры", "А.Семкин", "Thanks for watching", "продолжение следует"]

    # Собираем список файлов для текущей сессии, чтобы потом их удалить
    session_files = []

    try:
        for user_id, audio in sink.audio_data.items():
            # Уникальное имя файла: ID сервера + ID пользователя
            file_name = f"temp_{guild_id}_{user_id}.wav"
            session_files.append(file_name)
            
            audio.file.seek(0)
            data = audio.file.read()

            if len(data) < 2000:
                continue

            with open(file_name, "wb") as f:
                f.write(data)

            try:
                # Асинхронный запуск Whisper
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
                    if not any(junk.lower() in text.lower() for junk in junk_phrases):
                        all_phrases.append({
                            'time': segment['start'],
                            'user': username,
                            'text': text
                        })
            except Exception as e:
                print(f"❌ Ошибка Whisper ({guild_name}): {e}")

        # Сортировка и сборка текста
        all_phrases.sort(key=lambda x: x['time'])
        raw_transcript_lines = []
        for p in all_phrases:
            m, s = divmod(int(p['time']), 60)
            raw_transcript_lines.append(f"[{m:02d}:{s:02d}] **{p['user']}**: {p['text']}")

        raw_text = "\n".join(raw_transcript_lines)

        if not raw_text:
            await status_msg.edit(content="😶 Не удалось разобрать слова в этом канале.")
            return

        # Отправка стенограммы
        if len(raw_text) > 1900:
            with open(f"transcript_{guild_id}.txt", "w", encoding="utf-8") as f:
                f.write(raw_text.replace("**", ""))
            await channel.send("📋 Стенограмма:", file=discord.File(f"transcript_{guild_id}.txt"))
            if os.path.exists(f"transcript_{guild_id}.txt"): os.remove(f"transcript_{guild_id}.txt")
        else:
            await status_msg.edit(content=f"📋 **СТЕНОГРАММА ({guild_name})**\n\n{raw_text}")

        # Анализ через Ollama
        await channel.send("🧠 **Ватсон анализирует контекст...**")
        
        prompt = f"""
        Context: Discord server '{guild_name}'.
        Task: Summarize and find Action Items in Russian.
        Transcript:
        {raw_text}
        """
        
        ai_analysis = await ask_ollama(prompt)
        
        # Отправка анализа
        if len(ai_analysis) > 1900:
            with open(f"analysis_{guild_id}.txt", "w", encoding="utf-8") as f:
                f.write(ai_analysis)
            await channel.send("📝 Анализ:", file=discord.File(f"analysis_{guild_id}.txt"))
            if os.path.exists(f"analysis_{guild_id}.txt"): os.remove(f"analysis_{guild_id}.txt")
        else:
            await channel.send(f"📝 **АНАЛИЗ ({guild_name}):**\n\n{ai_analysis}")

    finally:
        # Гарантированное удаление временных аудиофайлов
        for f_path in session_files:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"✅ Обработка для {guild_name} завершена, файлы удалены.")

    
token = os.getenv("DISCORD_TOKEN")
if not token:
    raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example)")
bot.run(token)
