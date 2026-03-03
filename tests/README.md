# Watson tests

## Current tests

- **Mocks** — `conftest.py` mocks `discord`, `faster_whisper`, `ollama`, `psutil`; `main` is imported without a token or model.
- **Logic** — `build_transcript_lines`, recording limits, command rejections (`!record` when not in voice, or when transcription is in progress in the same guild).
- **Concurrent recordings** — `test_record_allowed_other_guild_while_one_transcribing` ensures the “transcription in progress” block applies per guild: guild B can start `!record` while guild A is transcribing.

Run from project root: `pytest tests/ -v`.

## Testing concurrent recordings in different channels

### 1. With mocks (no real Discord)

- Existing tests already check that different guilds do not block each other via `transcribing_guilds`.
- To go further: add a test that calls `once_done` for two guilds in parallel (`asyncio.gather`) with mocked sink (e.g. `audio_data = {user_id: MockAudio(bytes)}`), channel, and `model.transcribe` / file I/O, to confirm two concurrent `once_done` runs do not interfere (separate `temp_guild_dir`, separate `guild_id` in `transcribing_guilds`). No extra “bot” processes needed; use mocks only.

### 2. Manual run with real bot

- Two servers (or two voice channels; one bot can only be in one voice channel per server).
- In each: join voice, `!join`, `!record`. Leave the channel in one server — the bot should process only that recording and leave only that channel; the other keeps recording.
- This checks real Discord and voice behaviour without automation.
