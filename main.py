"""
Discord-бот Watson: запись голоса в канале и транскрипция через faster-whisper.

Сценарий работы:
  1. Пользователь в голосовом канале вызывает !join — бот подключается к каналу.
  2. !record — начинается запись; аудио пишется блоками (ChunkedWaveSink) по BLOCK_DURATION_SECONDS.
  3. !stop или выход всех из канала — запись останавливается, блоки отправляются в очередь транскрипции.
  4. Фоновый worker обрабатывает блоки Whisper'ом, склеивает результаты и постит транскрипт в канал.

Команды: !join, !leave, !record, !stop, !check.
Конфигурация: .env (DISCORD_TOKEN, RECORDING_MAX_MINUTES, BLOCK_DURATION_SECONDS, TRANSCRIPT_LANGUAGE и др.).
"""

import asyncio
import gc
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone

import psutil
import discord
from discord.ext import commands
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from watson_sink import ChunkedWaveSink

load_dotenv()

# --- Каталоги ---
# Корень проекта (каталог, где лежит main.py) — для относительных путей
_project_root = os.path.dirname(os.path.abspath(__file__))

# Временный каталог для WAV-блоков и рабочих файлов (не засоряем корень проекта)
_watson_temp_dir = os.getenv("WATSON_TEMP_DIR")
if not _watson_temp_dir:
    _watson_temp_dir = os.path.join(tempfile.gettempdir(), "watson")
os.makedirs(_watson_temp_dir, exist_ok=True)

# Каталог для сохранённых транскриптов и WAV: по умолчанию recordings/ в проекте
_watson_recordings_dir = os.getenv("WATSON_RECORDINGS_DIR")
if not _watson_recordings_dir:
    _watson_recordings_dir = os.path.join(_project_root, "recordings")
else:
    if not os.path.isabs(_watson_recordings_dir):
        _watson_recordings_dir = os.path.join(_project_root, _watson_recordings_dir)
os.makedirs(_watson_recordings_dir, exist_ok=True)

# --- Логирование ---
# Уровень из env (LOG_LEVEL), опционально вывод в файл (LOG_FILE)
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


def _memory_mb() -> float | None:
    """Текущее потребление памяти процесса (RSS) в МБ; None при ошибке."""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _log_memory(stage: str) -> None:
    """Пишет в лог потребление памяти на указанном этапе (для отладки)."""
    mb = _memory_mb()
    if mb is not None:
        logger.info("Memory [%s]: %.1f MB RSS", stage, mb)


def build_transcript_lines(phrases: list[dict]) -> str:
    """
    Собирает текст транскрипта из списка фраз.
    phrases: список dict с ключами time (секунды), user, text.
    Возвращает строку вида "[MM:SS] **User**: text\\n" для поста в Discord.
    """
    lines = []
    for p in phrases:
        m, s = divmod(int(p["time"]), 60)
        lines.append(f"[{m:02d}:{s:02d}] **{p['user']}**: {p['text']}\n")
    return "".join(lines)


# --- Снижение шума в логах Discord ---
# Ошибки декодирования Opus (потеря/повреждение пакетов) — не критичны, скрываем из лога
class _SuppressOpusDecodeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "decoding opus frame" not in (record.getMessage() or "").lower()

_root_logger = logging.getLogger()
_root_logger.addFilter(_SuppressOpusDecodeFilter())

logging.getLogger("discord.voicereader").setLevel(logging.CRITICAL)
logging.getLogger("discord.voicereader").propagate = False
logging.getLogger("discord.voice_client").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# --- Opus ---
# Нужен для декодирования голоса Discord; подгружаем до создания бота.
# Порядок: env OPUS_LIB_PATH → macOS (Homebrew/Intel) → Linux. При ошибках декодирования проверьте версию libopus.
_opus_paths = [
    os.getenv("OPUS_LIB_PATH"),
    "/opt/homebrew/lib/libopus.dylib",   # macOS Apple Silicon Homebrew
    "/usr/local/lib/libopus.dylib",       # macOS Intel Homebrew
    "libopus.so.0",                       # Linux (Debian/Ubuntu: libopus0)
]
_opus_loaded = False
for path in _opus_paths:
    if not path:
        continue
    try:
        discord.opus.load_opus(path)
        logger.info("Opus loaded: %s", path)
        _opus_loaded = True
        break
    except Exception:
        pass
if not _opus_loaded:
    logger.info("Opus: using library default")
_log_memory("after_opus")

# --- Модель Whisper ---
# turbo + cpu + int8 — компромисс скорость/качество/память
model = WhisperModel("turbo", device="cpu", compute_type="int8")
logger.info("Whisper ready")
logger.info("Temp dir for recordings: %s", _watson_temp_dir)
_log_memory("after_whisper_load")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True  # обязателен для голоса и on_voice_state_update

bot = commands.Bot(command_prefix='!', intents=intents)

# --- Глобальное состояние ---
# Гильдии, в которых идёт транскрипция (новую запись не начинаем, пока не закончится)
transcribing_guilds = set()
# Куда писать, если бота выкинуло из голоса: guild_id -> текстовый канал
last_text_channel_for_voice: dict[int, discord.TextChannel] = {}
# Гильдии, где только что вызвали !leave — не показываем "соединение разорвано"
left_via_command: set[int] = set()

# --- Параметры записи и транскрипции ---
MAX_RECORDING_MINUTES = int(os.getenv("RECORDING_MAX_MINUTES", "30"))
MAX_RECORDING_SECONDS = MAX_RECORDING_MINUTES * 60

_transcript_lang = (os.getenv("TRANSCRIPT_LANGUAGE") or "").strip()
TRANSCRIPT_LANGUAGE = _transcript_lang or None  # None = автоопределение языка
TRANSCRIPT_BEAM_SIZE = int(os.getenv("TRANSCRIPT_BEAM_SIZE", "5"))

# Треки короче этого (байт) не отдаём в Whisper
MIN_AUDIO_BYTES = 2000

# Длительность одного блока записи в секундах (например 300 = 5 мин)
BLOCK_DURATION_SECONDS = int(os.getenv("BLOCK_DURATION_SECONDS", "300"))

# Очередь заданий: (session_id, block_index, block_dict); воркер забирает и транскрибирует
transcribe_queue: asyncio.Queue = asyncio.Queue()
# Метаданные сессии: (guild_id, channel_id) -> { channel, timestamp, block_paths, results, ... }
session_meta: dict[tuple[int, int], dict] = {}
# Фразы-мусор от Whisper (типовые подписи) — выкидываем из транскрипта
_junk_phrases = ["editor", "subtitles", "thanks for watching", "to be continued", "а.семкин", "субтитры", "продолжение следует", "спасибо за просмотр"]


# Формат для Whisper: 16 kHz моно 16-bit (ускоряет и уменьшает размер)
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


def ensure_whisper_format(path: str) -> tuple[str, bool]:
    """
    Приводит WAV к формату 16 kHz моно 16-bit для Whisper.
    Возвращает (путь к файлу для использования, создан ли временный файл).
    Если конвертация не удалась, возвращает исходный path и False.
    """
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_wav(path)
        if seg.frame_rate == WHISPER_SAMPLE_RATE and seg.channels == WHISPER_CHANNELS:
            return path, False
        seg = seg.set_frame_rate(WHISPER_SAMPLE_RATE).set_channels(WHISPER_CHANNELS)
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="whisper_")
        os.close(fd)
        seg.export(temp_path, format="wav")
        return temp_path, True
    except Exception:
        return path, False


def _transcribe_one(path: str):
    """
    Транскрибирует один WAV файл через Whisper (блокирующий вызов).
    Ожидает 16 kHz моно или путь от ensure_whisper_format. Возвращает список segment-объектов.
    """
    segments_iter, _ = model.transcribe(path, beam_size=TRANSCRIPT_BEAM_SIZE, language=TRANSCRIPT_LANGUAGE)
    return list(segments_iter)


async def _merge_and_post(session_id: tuple[int, int]):
    """
    Сливает результаты всех блоков сессии в один транскрипт, сохраняет файлы,
    при необходимости склеивает WAV по пользователям и отправляет сообщения в канал.
    Вызывается воркером, когда для session_id готовы все блоки (len(results) == total_blocks).
    """
    meta = session_meta.get(session_id)
    if not meta or len(meta["results"]) != meta["total_blocks"]:
        return
    guild_id, channel_id = session_id
    channel = meta["channel"]
    logger.info("Merge: session %s merging %d blocks (guild %s)", session_id, meta["total_blocks"], guild_id)
    # Собираем фразы из всех блоков, сдвигая время на offset блока
    block_offset = BLOCK_DURATION_SECONDS
    all_phrases = []
    for block_index in range(meta["total_blocks"]):
        offset = block_index * block_offset
        for phrase in meta["results"].get(block_index, []):
            all_phrases.append({
                "time": offset + phrase["time"],
                "user": phrase["user"],
                "text": phrase["text"],
            })
    all_phrases.sort(key=lambda x: x["time"])
    raw_transcript = build_transcript_lines(all_phrases)
    if not raw_transcript.strip():
        logger.info("Merge: no speech recognized (session %s)", session_id)
        try:
            await channel.send("😶 Речь не распознана.")
        except discord.DiscordException:
            pass
        _cleanup_session(session_id, meta)
        return
    transcript_plain = raw_transcript.replace("**", "")
    timestamp = meta["timestamp"]
    safe_guild = meta["safe_guild"]
    safe_channel = meta["safe_channel"]
    transcript_path = os.path.join(
        _watson_recordings_dir,
        f"{timestamp}-{safe_guild}-{safe_channel}-transcript.txt",
    )
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_plain)
        logger.info("Merge: saved transcript %s (%d phrases)", transcript_path, len(all_phrases))
    except OSError as e:
        logger.warning("Merge: could not save transcript %s: %s", transcript_path, e)
    recording_paths = []
    try:
        from pydub import AudioSegment
        for user_id, paths in _block_paths_by_user(meta["block_paths"]).items():
            if not paths:
                continue
            combined = AudioSegment.empty()
            for p in paths:
                if os.path.exists(p):
                    combined += AudioSegment.from_wav(p)
            if len(combined) > 0:
                full_path = os.path.join(
                    _watson_recordings_dir,
                    f"{timestamp}-{safe_guild}-{safe_channel}-user{user_id}-full.wav",
                )
                combined.export(full_path, format="wav")
                recording_paths.append(full_path)
        if recording_paths:
            logger.info("Merge: saved %d WAV(s) for session %s", len(recording_paths), session_id)
    except Exception as e:
        logger.warning("Merge: could not concat WAVs: %s", e)
    try:
        header = "📋 **Транскрипт**\n\n"
        # Лимит сообщения Discord 2000 символов — длинный транскрипт уходим файлом
        if len(header) + len(raw_transcript) > 2000:
            await channel.send(header + "*(вложение)*", file=discord.File(transcript_path))
        else:
            await channel.send(header + raw_transcript)
        saved = [os.path.basename(p) for p in recording_paths]
        if transcript_path and os.path.exists(transcript_path):
            saved.append(os.path.basename(transcript_path))
        if saved:
            # Показываем полный путь к каталогу, чтобы пользователь знал, где искать файлы
            await channel.send(f"📁 Сохранено в каталог: `{_watson_recordings_dir}`\nФайлы: " + ", ".join(saved))
    except discord.DiscordException as e:
        logger.exception("Merge: failed to send (guild %s): %s", guild_id, e)
    _cleanup_session(session_id, meta)
    logger.info("Merge: session %s done (guild %s)", session_id, guild_id)


def _block_paths_by_user(block_paths: list[dict[int, str]]) -> dict[int, list[str]]:
    """
    Преобразует список блоков {user_id: path} в словарь user_id -> [path_block0, path_block1, ...]
    в порядке блоков (для склейки WAV по пользователям).
    """
    by_user: dict[int, list[str]] = {}
    for block in block_paths:
        for user_id, path in block.items():
            by_user.setdefault(user_id, []).append(path)
    return by_user


def _cleanup_session(session_id: tuple[int, int], meta: dict):
    """Удаляет временный каталог сессии, убирает сессию из session_meta и гильдию из transcribing_guilds."""
    guild_id = session_id[0]
    temp_dir = meta.get("temp_guild_dir")
    if temp_dir and os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            logger.warning("Could not remove temp dir %s: %s", temp_dir, e)
    session_meta.pop(session_id, None)
    transcribing_guilds.discard(guild_id)


async def transcription_worker():
    """
    Фоновый воркер: забирает из transcribe_queue задания (session_id, block_index, block_dict),
    транскрибирует каждый блок, по заполнению всех блоков сессии вызывает _merge_and_post.
    Запускается один раз в on_ready.
    """
    logger.info("Worker: transcription worker started")
    while True:
        try:
            session_id, block_index, block_dict = await transcribe_queue.get()
        except asyncio.CancelledError:
            logger.info("Worker: cancelled")
            break
        logger.info("Worker: job session=%s block=%d users=%d", session_id, block_index, len(block_dict))
        try:
            await _process_transcribe_job(session_id, block_index, block_dict)
        except Exception as e:
            logger.exception("Worker: job failed session=%s block=%s: %s", session_id, block_index, e)
        finally:
            transcribe_queue.task_done()


async def _process_transcribe_job(session_id: tuple[int, int], block_index: int, block_dict: dict):
    """
    Обрабатывает один блок: для каждого user_id в block_dict проверяет WAV, при необходимости
    конвертирует в 16 kHz моно, вызывает Whisper, собирает фразы (без junk_phrases).
    Результат пишет в meta["results"][block_index]; если накопились все блоки сессии — вызывает _merge_and_post.
    """
    meta = session_meta.get(session_id)
    if not meta:
        logger.warning("Worker: no session_meta for %s block %s, skip", session_id, block_index)
        return
    block_phrases = []
    try:
        for user_id, path in block_dict.items():
            if not os.path.exists(path) or os.path.getsize(path) < MIN_AUDIO_BYTES:
                logger.debug("Worker: skip user %s block %s (no file or too small)", user_id, block_index)
                continue
            path_to_use, was_temp = await asyncio.to_thread(ensure_whisper_format, path)
            try:
                segments_list = await asyncio.to_thread(_transcribe_one, path_to_use)
            except Exception as e:
                logger.exception("Worker: Whisper error session=%s block=%s user=%s: %s", session_id, block_index, user_id, e)
                continue
            finally:
                if was_temp and path_to_use != path and os.path.exists(path_to_use):
                    try:
                        os.remove(path_to_use)
                    except OSError:
                        pass
            user_obj = bot.get_user(user_id)
            username = user_obj.display_name if user_obj else f"User {user_id}"
            for seg in segments_list:
                text = (seg.text or "").strip()
                if not any(junk in text.lower() for junk in _junk_phrases) and len(text) > 1:
                    block_phrases.append({
                        "time": seg.start,
                        "user": username,
                        "text": text,
                    })
        logger.info("Worker: session=%s block=%d phrases=%d", session_id, block_index, len(block_phrases))
    finally:
        meta["results"][block_index] = block_phrases
        if len(meta["results"]) == meta["total_blocks"]:
            logger.info("Worker: session %s all blocks done, calling merge", session_id)
            await _merge_and_post(session_id)


@bot.event
async def on_ready() -> None:
    """Вызывается при успешном подключении бота к Discord; запускаем воркер транскрипции."""
    logger.info("Watson online — %s (ID: %s), guilds: %d", bot.user.name, bot.user.id, len(bot.guilds))
    _log_memory("on_ready")
    asyncio.create_task(transcription_worker())
    for guild in bot.guilds:
        logger.info("  Guild: %s (ID: %s)", guild.name, guild.id)


@bot.event
async def on_voice_state_update(member, before, after):
    """
    Обработка смены голосового состояния:
    - Если бот сам вышел из канала (сеть/Discord разорвали) — уведомляем в last_text_channel (если не !leave).
    - Если пользователь вышел из канала, где сидит бот, и людей не осталось — останавливаем запись и выходим.
    """
    if member.id == bot.user.id and before.channel is not None and after.channel is None:
        gid = member.guild.id
        logger.info("Voice: bot left channel %s (guild %s)", before.channel.name, gid)
        if gid in left_via_command:
            left_via_command.discard(gid)
            return
        ch = last_text_channel_for_voice.get(gid)
        if ch:
            try:
                await ch.send("🔌 Соединение разорвано. Используйте `!join` снова.")
            except discord.DiscordException:
                pass
        return
    if before.channel is None:
        return
    voice_client = member.guild.voice_client
    if not voice_client or voice_client.channel != before.channel:
        return
    # Считаем оставшихся людей в канале (member уже мог быть исключён из members в части версий API)
    humans_remaining = [m for m in before.channel.members if m != member and not m.bot]
    logger.info("Voice: %s left %s (guild %s), humans_remaining=%d", member.display_name, before.channel.name, member.guild.id, len(humans_remaining))
    if len(humans_remaining) != 0:
        return
    if voice_client.recording:
        logger.info("Voice: channel empty, stopping recording (guild %s)", member.guild.id)
        voice_client.stop_recording()
    await voice_client.disconnect()
    logger.info("Voice: left channel %s (guild %s), no users left", before.channel.name, member.guild.id)


@bot.command()
async def check(ctx):
    """Проверка: подключение бота, находитесь ли вы в голосе, права на сообщения/файлы в канале."""
    logger.info("Check: !check from %s in #%s (guild %s)", ctx.author, ctx.channel.name, ctx.guild.id)
    perms = ctx.channel.permissions_for(ctx.me)
    status = [
        f"Подключение: OK",
        f"Вы в голосе: {'да' if ctx.author.voice else 'нет'}",
        f"Сообщения/файлы: {'да' if perms.send_messages and perms.attach_files else 'нет'}",
    ]
    embed = discord.Embed(
        title="Watson",
        description="\n".join(status),
        color=discord.Color.blue() if perms.attach_files else discord.Color.red()
    )
    await ctx.send(embed=embed)


@bot.command()
async def join(ctx):
    """Подключиться к голосовому каналу, в котором находится автор команды."""
    logger.info("Voice: !join from %s in guild %s", ctx.author, ctx.guild.id)
    if ctx.voice_client:
        logger.info("Voice: already in channel %s (guild %s)", ctx.voice_client.channel.name, ctx.guild.id)
        return await ctx.send("Уже в канале. 🎙")
    if not ctx.author.voice:
        logger.info("Voice: reject join — author not in voice (guild %s)", ctx.guild.id)
        return await ctx.send("Сначала зайдите в голосовой канал.")
    ch = ctx.author.voice.channel
    last_text_channel_for_voice[ctx.guild.id] = ctx.channel
    logger.info("Voice: connecting to %s (guild %s)...", ch.name, ctx.guild.id)
    try:
        await ch.connect(timeout=60.0, reconnect=True)
    except asyncio.TimeoutError:
        logger.warning("Voice: connect timeout guild %s", ctx.guild.id)
        await ctx.send("⚠️ Таймаут подключения. Сделайте `!leave`, подождите 5–10 сек, затем `!join`.")
        return
    except discord.DiscordException as e:
        logger.exception("Voice: connect failed guild %s: %s", ctx.guild.id, e)
        await ctx.send("⚠️ Не удалось подключиться. Проверьте сеть и попробуйте снова.")
        return
    logger.info("Voice: connected to %s (guild %s)", ch.name, ctx.guild.id)
    await ctx.send("🎩 В канале.")


async def _enforce_recording_limit(guild_id: int, channel_id: int):
    """
    Фоновая задача: через MAX_RECORDING_SECONDS останавливает текущую запись и пишет в канал.
    Создаётся при старте записи (!record); проверяет, что гильдия и voice ещё существуют.
    """
    await asyncio.sleep(MAX_RECORDING_SECONDS)
    guild = bot.get_guild(guild_id)
    if not guild:
        logger.debug("Record: limit timer — guild %s gone", guild_id)
        return
    voice = guild.voice_client
    if voice and voice.recording:
        logger.info("Record: limit %d min reached (guild %s), stopping", MAX_RECORDING_MINUTES, guild_id)
        voice.stop_recording()
        ch = guild.get_channel(channel_id)
        if ch:
            try:
                await ch.send(f"⏱ Лимит {MAX_RECORDING_MINUTES} мин. Для новой записи: `!record`.")
            except discord.DiscordException:
                pass


@bot.command()
async def record(ctx):
    """Начать запись голоса в текущем канале; макс. длительность задаётся RECORDING_MAX_MINUTES (по умолчанию 30 мин)."""
    logger.info("Record: !record from %s in guild %s (#%s)", ctx.author, ctx.guild.id, ctx.channel.name)
    voice = ctx.voice_client
    if not voice:
        logger.info("Record: reject — bot not in voice (guild %s)", ctx.guild.id)
        return await ctx.send("Сначала пригласите бота: `!join`.")
    if voice.recording:
        logger.info("Record: reject — already recording (guild %s)", ctx.guild.id)
        return await ctx.send("Запись уже идёт.")
    if ctx.guild.id in transcribing_guilds:
        logger.info("Record: reject — transcription in progress (guild %s)", ctx.guild.id)
        return await ctx.send("Идёт обработка предыдущей записи. Дождитесь окончания.")

    # Голосовой клиент может существовать до завершения handshake; ждём is_connected до 30 сек
    if not voice.is_connected():
        logger.info("Record: waiting for voice handshake (guild %s)", ctx.guild.id)
        await ctx.send("⏳ Подключение к голосу…")
        for i in range(60):
            await asyncio.sleep(0.5)
            if voice.is_connected():
                logger.info("Record: voice handshake OK after %.1fs (guild %s)", i * 0.5, ctx.guild.id)
                break
            voice = ctx.guild.voice_client
            if not voice:
                logger.warning("Record: voice client lost while waiting (guild %s)", ctx.guild.id)
                return await ctx.send("⚠️ Соединение потеряно. Используйте `!join` снова.")
        if not voice.is_connected():
            try:
                await voice.disconnect(force=True)
            except Exception:
                pass
            logger.warning("Record: voice handshake timeout 30s (guild %s)", ctx.guild.id)
            return await ctx.send("⚠️ Голос не подключился за 30 с. `!leave` → подождите 5–10 сек → `!join` → `!record`.")

    last_text_channel_for_voice[ctx.guild.id] = ctx.channel
    temp_guild_dir = os.path.join(_watson_temp_dir, str(ctx.guild.id))
    os.makedirs(temp_guild_dir, exist_ok=True)
    # Sink пишет аудио блоками по BLOCK_DURATION_SECONDS; по завершении записи вызывается once_done_chunked
    sink = ChunkedWaveSink(
        temp_guild_dir=temp_guild_dir,
        guild_id=ctx.guild.id,
        block_duration_seconds=BLOCK_DURATION_SECONDS,
    )
    try:
        voice.start_recording(sink, once_done_chunked, ctx.channel)
    except Exception as e:
        logger.exception("Record: start_recording failed guild %s: %s", ctx.guild.id, e)
        await ctx.send("⚠️ Не удалось начать запись. См. логи.")
        return
    logger.info("Record: started in %s (guild %s), max %d min, block %d s", voice.channel.name, ctx.guild.id, MAX_RECORDING_MINUTES, BLOCK_DURATION_SECONDS)
    await ctx.send(f"⏺ Запись (макс. {MAX_RECORDING_MINUTES} мин). Остановка: `!stop`.")
    asyncio.create_task(_enforce_recording_limit(ctx.guild.id, ctx.channel.id))


@bot.command()
async def stop(ctx):
    """Остановить текущую запись; блоки отправляются в очередь транскрипции, результат придёт в канал."""
    logger.info("Record: !stop from %s in guild %s", ctx.author, ctx.guild.id)
    voice = ctx.voice_client
    if voice and voice.recording:
        logger.info("Record: stopping in %s (guild %s)", voice.channel.name, ctx.guild.id)
        voice.stop_recording()
        await ctx.send("⏹ Остановлено.")
    else:
        logger.info("Record: reject stop — not recording (guild %s)", ctx.guild.id)
        await ctx.send("Сейчас не записываю.")


@bot.command()
async def leave(ctx):
    """Выйти из голосового канала."""
    logger.info("Voice: !leave from %s in guild %s", ctx.author, ctx.guild.id)
    if ctx.voice_client:
        ch_name = ctx.voice_client.channel.name
        left_via_command.add(ctx.guild.id)
        await ctx.voice_client.disconnect()
        logger.info("Voice: left %s (guild %s)", ch_name, ctx.guild.id)
        await ctx.send("Пока.")
    else:
        logger.info("Voice: reject leave — not in channel (guild %s)", ctx.guild.id)
        await ctx.send("Не в голосовом канале.")


async def once_done_chunked(sink, channel: discord.TextChannel, *args) -> None:
    """
    Колбэк, вызываемый после остановки записи (stop_recording или конец по таймеру/выход).
    Читает sink.blocks (список блоков {user_id: path}), создаёт сессию в session_meta,
    кладёт каждый блок в transcribe_queue и добавляет гильдию в transcribing_guilds.
    """
    blocks = getattr(sink, "blocks", None)
    guild_id = channel.guild.id
    if not blocks:
        logger.warning("Record: recording empty, no blocks (guild %s)", guild_id)
        temp_dir = getattr(sink, "_temp_guild_dir", None)
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass
        try:
            await channel.send("📭 Аудио не получено. Проверьте соединение, затем `!leave` → `!join` → `!record`.")
        except discord.DiscordException:
            pass
        return
    logger.info("Record: done, blocks=%d (guild %s), enqueueing", len(blocks), guild_id)
    session_id = (guild_id, channel.id)
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        guild_name = channel.guild.name
        # Безопасные имена для имён файлов (только буквы, цифры, дефис, подчёркивание)
        safe_guild = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in guild_name)
        safe_channel = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in channel.name)
        session_meta[session_id] = {
            "channel": channel,
            "guild_name": guild_name,
            "timestamp": timestamp,
            "safe_guild": safe_guild,
            "safe_channel": safe_channel,
            "total_blocks": len(blocks),
            "block_paths": list(blocks),
            "results": {},
            "temp_guild_dir": getattr(sink, "_temp_guild_dir", os.path.join(_watson_temp_dir, str(guild_id))),
        }
        for block_index, block_dict in enumerate(blocks):
            await transcribe_queue.put((session_id, block_index, block_dict))
        transcribing_guilds.add(guild_id)
        logger.info("Record: session %s enqueued %d blocks (guild %s)", session_id, len(blocks), guild_id)
        try:
            await channel.send("⏳ Обработка аудио…")
        except discord.DiscordException:
            pass
    except Exception as e:
        logger.exception("Record: once_done_chunked failed guild %s: %s", guild_id, e)
        transcribing_guilds.discard(guild_id)
        session_meta.pop(session_id, None)
        # Удаляем временный каталог с WAV-блоками (при ошибке _cleanup_session не вызывается)
        temp_dir = getattr(sink, "_temp_guild_dir", None)
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Record: removed temp dir %s after once_done error", temp_dir)
            except OSError as err:
                logger.warning("Record: could not remove temp dir %s: %s", temp_dir, err)
        try:
            await channel.send("⚠️ Ошибка обработки. Попробуйте `!record` снова.")
        except discord.DiscordException:
            pass


# --- Запуск ---
token = os.getenv("DISCORD_TOKEN")
if not token:
    logger.error("DISCORD_TOKEN not found in .env")
    raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example)")
logger.info("Starting bot...")
bot.run(token)
