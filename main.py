import os
import asyncio
import discord
import whisper
from discord.ext import commands
from dotenv import load_dotenv

# 1. Настройка окружения
load_dotenv()

# Загрузка Opus (необходимо для некоторых серверных ОС и macOS)
try:
    # Если на сервере Linux, путь может не требоваться или быть другим
    # discord.opus.load_opus('/usr/lib/libopus.so') 
    discord.opus.load_opus('/opt/homebrew/lib/libopus.dylib')
    print("✅ Opus loaded")
except Exception as e:
    print(f"ℹ️ Opus load info (standard path used): {e}")

# 2. Инициализация Whisper
print("Loading Whisper model (turbo)...")
# На сервере без GPU автоматически выберет CPU. 
# Если есть GPU NVIDIA, Whisper сам задействует CUDA.
model = whisper.load_model("turbo")
print("✅ Whisper ready.")

# 3. Конфигурация бота
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'--- Watson Online ---')
    print(f'Bot: {bot.user.name} (ID: {bot.user.id})')
    print(f'Connected to {len(bot.guilds)} guilds.')
    print('----------------------')

@bot.event
async def on_voice_state_update(member, before, after):
    """Надежная автоматическая остановка записи."""
    # Нас интересует только когда кто-то уходит (before.channel)
    if before.channel is not None:
        voice_client = member.guild.voice_client
        
        # Если бот в канале, где произошло движение
        if voice_client and voice_client.channel == before.channel:
            # Считаем людей (исключая ботов)
            human_members = [m for m in before.channel.members if not m.bot]
            
            if len(human_members) == 0:
                if voice_client.recording:
                    print(f"🤫 Канал {before.channel.name} пуст. Завершаю сессию...")
                    
                    # Принудительно вызываем остановку
                    voice_client.stop_recording()
                    
                    # Даем небольшую паузу, чтобы once_done успел подхватить данные 
                    # до того, как бот выйдет из канала (если планируется выход)
                    await asyncio.sleep(1)

@bot.command()
async def check(ctx):
    """Проверка прав бота в текущем канале."""
    perms = ctx.channel.permissions_for(ctx.me)
    
    status = [
        f"✅ **Подключение:** Стабильное",
        f"🎤 **Голосовой канал:** {'✅' if ctx.author.voice else '❌ (вы не в канале)'}",
        f"📝 **Отправка сообщений:** {'✅' if perms.send_messages else '❌'}",
        f"📎 **Отправка файлов:** {'✅' if perms.attach_files else '❌'}",
        f"📜 **История сообщений:** {'✅' if perms.read_message_history else '❌'}",
        f"🎙 **Право записи (Speak):** {'✅' if perms.speak else '❌'}"
    ]
    
    embed = discord.Embed(
        title="Диагностика системы Ватсон",
        description="\n".join(status),
        color=discord.Color.blue() if perms.attach_files else discord.Color.red()
    )
    await ctx.send(embed=embed)

@bot.command()
async def join(ctx):
    if ctx.voice_client:
        return await ctx.send("Я уже в канале! 🎙")
    if ctx.author.voice:
        await ctx.author.voice.channel.connect()
        await ctx.send("🎩 Зашел. Готов к работе.")
    else:
        await ctx.send("Сначала сами зайдите в голосовой канал.")

@bot.command()
async def record(ctx):
    voice = ctx.voice_client
    if not voice:
        return await ctx.send("Сначала позовите меня командой !join")
    
    # ПРОВЕРКА: Если запись уже идет, не запускаем вторую
    if voice.recording:
        return await ctx.send("⚠️ Запись уже вовсю идет!")

    await ctx.send("⏺ **Запись пошла.**")
    voice.start_recording(discord.sinks.WaveSink(), once_done, ctx.channel)

@bot.command()
async def stop(ctx):
    """Остановить запись."""
    voice = ctx.voice_client
    if voice and voice.recording:
        voice.stop_recording()
        await ctx.send("⏹ Запись остановлена. Начинаю сборку текста...")
    else:
        await ctx.send("Я сейчас ничего не записываю.")

@bot.command()
async def leave(ctx):
    """Выйти из канала."""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("До встречи!")
    else:
        await ctx.send("Я не в голосовом канале.")

async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    """Обработка результатов записи."""
    if not sink.audio_data:
        await channel.send("📭 Запись пуста (тишина).")
        return

    guild_id = channel.guild.id
    status_msg = await channel.send("⚙️ **Ватсон обрабатывает аудио...**")
    
    all_phrases = []
    # Список фраз, которые Whisper часто выдумывает в тишине
    junk_phrases = ["редактор", "субтитры", "а.семкин", "продолжение следует", "спасибо за просмотр"]
    
    temp_files = []

    try:
        # Обработка аудио каждого участника
        for user_id, audio in sink.audio_data.items():
            # Уникальное имя файла для предотвращения конфликтов между серверами
            file_name = f"temp_{guild_id}_{user_id}.wav"
            temp_files.append(file_name)
            
            audio.file.seek(0)
            data = audio.file.read()

            if len(data) < 2000: # Пропускаем слишком короткие фрагменты
                continue

            with open(file_name, "wb") as f:
                f.write(data)

            try:
                # Транскрибация в отдельном потоке, чтобы не блокировать бота
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
                    # Фильтрация мусора и коротких артефактов
                    if not any(junk in text.lower() for junk in junk_phrases) and len(text) > 1:
                        all_phrases.append({
                            'time': segment['start'],
                            'user': username,
                            'text': text
                        })
            except Exception as e:
                print(f"❌ Ошибка Whisper для {user_id}: {e}")

        # Сортировка всех реплик по времени начала
        all_phrases.sort(key=lambda x: x['time'])
        
        raw_transcript = ""
        for p in all_phrases:
            m, s = divmod(int(p['time']), 60)
            raw_transcript += f"[{m:02d}:{s:02d}] **{p['user']}**: {p['text']}\n"

        if not raw_transcript:
            await status_msg.edit(content="😶 Не удалось распознать речь.")
            return

        # Отправка результата в Discord
        header = f"📋 **СТЕНОГРАММА ({channel.guild.name})**\n\n"
        
        if len(header + raw_transcript) > 2000:
            file_path = f"transcript_{guild_id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(raw_transcript.replace("**", "")) # Убираем жирный шрифт для файла
            
            await channel.send(header + "Результат во вложении:", file=discord.File(file_path))
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            await status_msg.edit(content=header + raw_transcript)

    finally:
        # Гарантированная очистка временных аудиофайлов
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        print(f"✅ Сессия завершена для сервера {guild_id}")

# Запуск
token = os.getenv("DISCORD_TOKEN")
if not token:
    print("❌ Ошибка: DISCORD_TOKEN не найден в .env")
else:
    bot.run(token)