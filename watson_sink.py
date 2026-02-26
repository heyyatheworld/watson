"""
Sink для поточной записи голоса блоками (chunked WAV).

Используется py-cord: голосовой поток приходит в write(data, user) кусками PCM.
ChunkedWaveSink накапливает данные по пользователям в _current[user_id] = BytesIO.
Каждые block_duration_seconds секунд таймер вызывает _rotate(): буферы сбрасываются
в WAV-файлы (по одному на пользователя на блок), буферы очищаются, блок добавляется
в self.blocks. При cleanup() (после stop_recording) вызывается последний _rotate(),
чтобы записать финальный блок без потери данных.

Формат выходных WAV: 16 kHz моно 16-bit, чтобы сразу отдавать в Whisper без конвертации в main.
"""

import asyncio
import io
import logging
import os
import threading
import wave

from discord.sinks.core import Filters, Sink, default_filters

logger = logging.getLogger(__name__)

# --- Форматы аудио ---
# Типичный вывод декодера Opus в py-cord: стерео 48 kHz 16-bit
WAV_CHANNELS = 2
WAV_SAMPLE_WIDTH = 2  # 16-bit
WAV_FRAMERATE = 48000

# Целевой формат для Whisper и для сохранения блоков (меньше размер, не нужна доп. конвертация в main)
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1
WHISPER_SAMPLE_WIDTH = 2


def convert_to_whisper_format(
    pcm_bytes: bytes,
    *,
    channels: int,
    sample_width: int,
    framerate: int,
) -> tuple[bytes, int, int, int]:
    """
    Конвертирует сырой PCM в 16 kHz моно 16-bit для Whisper.
    Возвращает (pcm_bytes, channels, sample_width, framerate) в целевом формате.
    При ошибке конвертации возвращает исходные параметры (fallback).
    """
    if channels == WHISPER_CHANNELS and framerate == WHISPER_SAMPLE_RATE and sample_width == WHISPER_SAMPLE_WIDTH:
        return pcm_bytes, WHISPER_CHANNELS, WHISPER_SAMPLE_WIDTH, WHISPER_SAMPLE_RATE
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_raw(
            io.BytesIO(pcm_bytes),
            sample_width=sample_width,
            frame_rate=framerate,
            channels=channels,
        )
        seg = seg.set_frame_rate(WHISPER_SAMPLE_RATE).set_channels(WHISPER_CHANNELS)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        with wave.open(buf, "rb") as wav_in:
            out_bytes = wav_in.readframes(wav_in.getnframes())
        return out_bytes, WHISPER_CHANNELS, WHISPER_SAMPLE_WIDTH, WHISPER_SAMPLE_RATE
    except Exception:
        return pcm_bytes, channels, sample_width, framerate


class ChunkedWaveSink(Sink):
    """
    Sink для записи голоса блоками по времени (например 5–10 мин).
    Запись непрерывная: между блоками нет пауз, буфер просто сбрасывается в файлы по таймеру.
    """

    def __init__(
        self,
        *,
        temp_guild_dir: str,
        guild_id: int,
        block_duration_seconds: int = 300,
        filters=None,
    ):
        if filters is None:
            filters = default_filters
        self.filters = filters
        Filters.__init__(self, **self.filters)
        self.encoding = "wav"
        self.vc = None  # задаётся в init(vc)
        self.audio_data = {}  # базовый Sink использует; мы пишем в _current
        self._temp_guild_dir = temp_guild_dir
        self._guild_id = guild_id
        self._block_duration_seconds = block_duration_seconds
        self._lock = threading.Lock()  # защита _current и blocks при записи из потока голоса
        self._current: dict[int, io.BytesIO] = {}  # user_id -> буфер PCM текущего блока
        self._block_index = 0
        self.blocks: list[dict[int, str]] = []  # готовые блоки: [{user_id: path}, ...]
        self.finished = False
        self._timer_task = None  # asyncio-задача таймера ротации

    def init(self, vc) -> None:
        """Вызывается py-cord при старте записи. Запускаем таймер ротации блоков в event loop голоса."""
        self.vc = vc
        super().init(vc)
        loop = getattr(vc, "loop", None)
        if loop is not None:
            def start_timer():
                self._timer_task = asyncio.ensure_future(self._timer_loop(), loop=loop)
                logger.info("Sink: timer started guild %s, block=%ds", self._guild_id, self._block_duration_seconds)
            loop.call_soon_threadsafe(start_timer)
        else:
            logger.warning("Sink: no event loop on voice client (guild %s), only final block will be saved", self._guild_id)

    def _write_wav(self, path: str, pcm_bytes: bytes, channels: int, sample_width: int, framerate: int) -> None:
        """Пишет PCM в WAV-файл с заданными параметрами."""
        with open(path, "wb") as f:
            with wave.open(f, "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(sample_width)
                w.setframerate(framerate)
                w.writeframes(pcm_bytes)

    def _rotate(self) -> None:
        """
        Сбрасывает текущие буферы в WAV-файлы (один файл на пользователя) и добавляет блок в self.blocks.
        Вызывается по таймеру из _timer_loop и в cleanup() для последнего неполного блока.
        Важно: _rotate() в cleanup() вызываем до установки self.finished = True, иначе ранний return.
        """
        with self._lock:
            if self.finished:
                return
            to_write: dict[int, bytes] = {}
            for user_id, buf in self._current.items():
                pcm = buf.getvalue()
                if len(pcm) > 0:
                    to_write[user_id] = pcm
                buf.seek(0)
                buf.truncate(0)
            block_index = self._block_index
            self._block_index += 1
        if not to_write:
            logger.debug("Sink: _rotate block %s empty (guild %s)", block_index, self._guild_id)
            return
        logger.info("Sink: _rotate block %s guild %s users=%d", block_index, self._guild_id, len(to_write))
        try:
            # Параметры декодера голоса (часто 48k stereo 16-bit)
            decoder = getattr(self.vc, "decoder", None) if self.vc else None
            if decoder is None:
                channels, sample_width, framerate = WAV_CHANNELS, WAV_SAMPLE_WIDTH, WAV_FRAMERATE
            else:
                channels = getattr(decoder, "CHANNELS", WAV_CHANNELS)
                sample_width = getattr(decoder, "SAMPLE_SIZE", WAV_SAMPLE_WIDTH * channels) // max(1, channels)
                framerate = getattr(decoder, "SAMPLING_RATE", WAV_FRAMERATE)
            block_paths: dict[int, str] = {}
            for user_id, pcm_bytes in to_write.items():
                pcm_out, out_ch, out_sw, out_fr = convert_to_whisper_format(
                    pcm_bytes, channels=channels, sample_width=sample_width, framerate=framerate,
                )
                path = os.path.join(
                    self._temp_guild_dir,
                    f"block_{block_index}_{user_id}.wav",
                )
                self._write_wav(path, pcm_out, out_ch, out_sw, out_fr)
                block_paths[user_id] = path
            with self._lock:
                self.blocks.append(block_paths)
        except Exception as e:
            logger.exception("Sink: _rotate failed block %s guild %s: %s", block_index, self._guild_id, e)

    async def _timer_loop(self) -> None:
        """
        Каждые block_duration_seconds секунд вызывает _rotate() в пуле потоков.
        _rotate() выполняет pydub-конвертацию и запись на диск — в основном потоке это
        может блокировать event loop и приводить к обрывам голоса (например 4006).
        """
        while not self.finished:
            await asyncio.sleep(self._block_duration_seconds)
            if self.finished:
                break
            await asyncio.to_thread(self._rotate)

    @Filters.container
    def write(self, data, user) -> None:
        """Вызывается из потока голоса py-cord для каждого куска PCM; накапливаем по user_id."""
        with self._lock:
            if user not in self._current:
                self._current[user] = io.BytesIO()
            self._current[user].write(data)

    def cleanup(self) -> None:
        """Вызывается при остановке записи. Отменяем таймер, сбрасываем последний блок, помечаем finished."""
        logger.info("Sink: cleanup guild %s blocks_so_far=%d", self._guild_id, len(self.blocks))
        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
        try:
            self._rotate()  # сброс последнего неполного блока до установки finished
        except Exception as e:
            logger.exception("Sink: cleanup _rotate failed guild %s: %s", self._guild_id, e)
        self.finished = True
