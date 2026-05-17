# Watson tests

## What runs in CI

- **`conftest.py`** — Injects fakes for `discord`, `faster_whisper`, and `ollama`, patches env, then reloads `watson.config`, `watson.permissions`, `watson.transcribe`, and `watson.handlers` so tests hit real handler code without a token or Whisper weights.
- **`test_main.py`** — `build_transcript_lines`, recording limits, and `!record` guard rails (not in voice, same-guild transcription lock, cross-guild allowed).
- **`test_voice_4017.py`** — `!join` survives a voice `ConnectionClosed` with code `4017` (DAVE / gateway rejection) and replies in-channel.

From the repo root:

```bash
pytest tests/ -v
```

## Going further (optional)

- **Parallel `once_done`** — Add a test that runs two mocked `once_done` flows with `asyncio.gather`, distinct `guild_id` temp dirs, and a stub sink, to assert the global transcription semaphore and `transcribing_guilds` behave as expected under overlap.
- **Manual multi-guild** — Two servers (or two guilds): `!join` / `!record` in each; verify leaving one channel only stops that session. Requires a real bot token and Discord.
