# Watson

Discord bot that records voice channel audio, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and optionally generates a short recap with [Ollama](https://ollama.com). Saves WAV files and transcript text to disk; posts only a recap (if enabled) and links to saved files in the channel.

**[Add Watson to your server](https://discord.com/oauth2/authorize?client_id=1474417737659846727&permissions=3147776&scope=bot)**

## Features

- **Voice recording** — Joins a voice channel and records participants (WAV). Max length and “warning before stop” are configurable.
- **Transcription** — Speech-to-text via faster-whisper (model/device/compute_type in `.env`). Runs in a thread so the bot stays responsive.
- **Saved files** — WAV and transcript `.txt` are written to a recordings directory (configurable). The bot does **not** post transcript text in the channel, only a recap (if Ollama is on) and paths to the saved files.
- **Ollama recap** — Optional short summary (200–300 chars) after each recording: what was discussed, decisions, who’s responsible. In the same language as the dialogue. Prompt is in `prompts/recap.txt`.
- **Auto-stop** — When the last human **leaves** the voice channel, recording stops and processing runs. Mute/deafen in the same channel is ignored (bot does not leave).
- **Config** — All settings via `.env` (prefix, temp/recordings dirs, Whisper, Ollama, recap prompt path). See `.env.example`.

## Prerequisites

- Python 3.10+
- [Discord bot token](https://discord.com/developers/applications) — create an application and bot; enable **Message Content Intent** and **Server Members Intent**
- **macOS (Homebrew):** Opus from `/opt/homebrew/lib/libopus.dylib` (or set `OPUS_LIB_PATH` in `.env`)

For **recap**: [Ollama](https://ollama.com) installed and running (e.g. `ollama run llama3.2`). Set `OLLAMA_RECAP_MODEL` in `.env` to enable.

## Adding Watson to your server

1. Open the **invite link** at the top of this README.
2. Select the server (you need **Manage Server** or **Administrator**).
3. Click **Authorize**.

Then use `!join`, `!record`, `!stop`, `!leave`, `!check` in a text channel (see **Usage**).

## Setup

1. **Clone and venv**

   ```bash
   cd Watson
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Main deps: `py-cord`, `faster-whisper`, `ollama`, `python-dotenv`, `psutil`.

3. **Configure**

   ```bash
   cp .env.example .env
   ```

   Set at least:

   ```env
   DISCORD_TOKEN=your_bot_token_here
   ```

   Optional (see `.env.example`):

   - `BOT_COMMAND_PREFIX` (default `!`)
   - `WATSON_TEMP_DIR`, `WATSON_RECORDINGS_DIR` (default `./temp`, `./recordings`)
   - `RECORDING_MAX_MINUTES`, `WARNING_BEFORE_STOP_MINUTES`
   - `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `TRANSCRIPT_LANGUAGE`, `TRANSCRIPT_BEAM_SIZE`
   - `OLLAMA_HOST`, `OLLAMA_RECAP_MODEL`, `RECAP_PROMPT_FILE` (recap prompt path)

   Bot and invite: [Developer Portal](https://discord.com/developers/applications) → Bot → enable intents → OAuth2 URL Generator (scope **bot**, permissions: View Channels, Connect, Speak, Send Messages, Read Message History, Attach Files).

## Usage

1. Invite the bot. Required permissions: **View Channels**, **Connect**, **Speak**, **Send Messages**, **Read Message History**, **Attach Files**.

2. In a text channel:

   | Command   | Description |
   | --------- | ----------- |
   | `!join`   | Bot joins your current voice channel |
   | `!record` | Start recording (max length from `RECORDING_MAX_MINUTES`; warning in channel before auto-stop) |
   | `!stop`   | Stop recording and process (transcribe + optional recap, then post recap and file links) |
   | `!leave`  | Bot leaves the voice channel |
   | `!check`  | Connection status and bot permissions (embed) |

3. After processing, the bot posts **Done**, then the **recap** (if `OLLAMA_RECAP_MODEL` is set), then **links to files** in the recordings directory (WAV per user, one transcript `.txt`). No full transcript text is sent in the channel.

**Saved transcript file** format: first line = header (date, time, guild name, channel name); blank line; recap (if any); blank line; transcript body.

## Docker

- **Ollama** runs as a separate service; the bot connects to it via `OLLAMA_HOST`.
- **Recordings** directory in the container is mounted from the host so files persist.

```bash
docker compose up -d
```

Then pull a model for recap (if you use it):

```bash
docker compose exec ollama ollama pull llama3.2
```

In `.env` set `OLLAMA_RECAP_MODEL=llama3.2` (and optionally override `OLLAMA_HOST`; compose sets `OLLAMA_HOST=http://ollama:11434`).

On a **remote host**, mount your host folders in `docker-compose.yml`, e.g.:

```yaml
volumes:
  - /data/watson/recordings:/app/recordings
  - /data/watson/temp:/app/temp
```

## Troubleshooting

- **Bot left when I muted** — Fixed: the bot only leaves when someone actually leaves the channel; mute/deafen in the same channel is ignored.
- **High memory** — Use `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8`; bot logs RSS at key stages.
- **Slow transcription** — Use GPU: `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16` (and install CUDA deps).
- **No recap** — Ensure Ollama is running and `OLLAMA_RECAP_MODEL` is set; in Docker, `OLLAMA_HOST=http://ollama:11434` is set by compose.
- **Bot doesn’t respond** — Enable **Message Content Intent** (and **Server Members Intent**) in the Developer Portal.

## Testing

Discord and faster-whisper are mocked; no token or model needed:

```bash
pytest tests/ -v
```

See `tests/README.md` for testing concurrent recordings (mocks and manual).

## Project structure

```
Watson/
├── main.py              # Bot, recording, Whisper, Ollama recap
├── prompts/
│   └── recap.txt        # Prompt for recap ({{TRANSCRIPT}} placeholder)
├── tests/
│   ├── conftest.py      # Mocks for import without Discord/Whisper
│   ├── test_main.py     # Helpers, config, command behaviour
│   └── README.md        # How to test concurrent recordings
├── Dockerfile           # Bot image (Ollama is separate in compose)
├── docker-compose.yml   # watson-bot + ollama, volume mounts
├── .env.example        # Env template
├── requirements.txt
└── README.md
```

## License

Use and modify as you like.
