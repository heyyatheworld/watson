# Watson

A Discord bot that records voice channel audio and transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). It posts timestamped transcripts in the channel and stops recording automatically when everyone leaves the voice channel.

**[Add Watson to your server](https://discord.com/oauth2/authorize?client_id=1474417737659846727&permissions=3147776&scope=bot)**

## Features

- **Voice recording** — Joins a voice channel and records participants (WAV).
- **Transcription** — Converts speech to text via faster-whisper (turbo model, CPU/int8); transcription runs in a thread so the bot stays responsive.
- **Timestamped transcript** — Posts a transcript with `[MM:SS] User: text` in the channel, or sends it as a file if longer than Discord’s limit.
- **Auto-stop** — When the last human leaves the voice channel, recording stops and the transcript is processed.
- **Logging** — Configurable log level and optional file output via environment variables.
- **Memory diagnostics** — Process RSS logged at key stages (after load, transcription start/done); optional `LOG_FILE` for persistence.

## Prerequisites

- Python 3.10+
- [Discord bot token](https://discord.com/developers/applications) — create an application and bot, enable **Message Content Intent** and **Server Members Intent**
- **macOS (Homebrew):** Opus is loaded from `/opt/homebrew/lib/libopus.dylib`; run `brew install opus` if needed

## Adding Watson to your server

1. Open the **invite link** at the top of this README in your browser.
2. Log in to Discord if prompted.
3. Select the server you want to add the bot to. You must have **Manage Server** or **Administrator** on that server.
4. Click **Authorize** and complete the captcha if shown.

Watson will appear in your server’s member list. You can then use `!join`, `!record`, `!stop` in a text channel (see **Usage** below).

## Setup

1. **Clone and create a virtual environment**

   ```bash
   cd Watson
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies: `py-cord`, `faster-whisper`, `python-dotenv`, `psutil`.

3. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:

   ```env
   DISCORD_TOKEN=your_bot_token_here
   ```

   Optional:

   ```env
   LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
   LOG_FILE=watson.log     # if set, logs are also written to this file
   ```

   To create the bot and get the token: [Developer Portal](https://discord.com/developers/applications) → New Application → Bot → enable **Server Members Intent** and **Message Content Intent** → copy Token. To give others an invite link: OAuth2 → URL Generator → scope **bot**, permissions View Channels, Connect, Speak, Send Messages, Read Message History, Attach Files → share the generated URL.

## Usage

1. Invite the bot to your server. Required permissions: **View Channels**, **Connect**, **Speak**, **Send Messages**, **Read Message History**, **Attach Files**.
2. In a text channel:

   | Command   | Description                                      |
   | --------- | ------------------------------------------------- |
   | `!join`   | Bot joins your current voice channel              |
   | `!record` | Start recording; speak in turn for best results   |
   | `!stop`   | Stop recording and get the transcript             |
   | `!leave`  | Bot leaves the voice channel                      |
   | `!check`  | Show connection status and bot permissions (embed)|

3. After `!stop`, the bot processes the audio with Whisper and posts the transcript. If the transcript is longer than 2000 characters, it is sent as `transcript_<guild_id>.txt`.

**Note:** Transcription uses `language="ru"`. You can change it in `main.py` in the `model.transcribe(...)` call. Model and device are set in code (`WhisperModel("turbo", device="cpu", compute_type="int8")`); use GPU by changing `device="cuda"` if available.

## Troubleshooting

- **High memory usage** — The bot logs RSS at key stages (`LOG_LEVEL=INFO`). After each session it runs `gc.collect()`. If you run on a small VPS, keep `device="cpu"` and `compute_type="int8"`.
- **Slow transcription** — On a machine with a CUDA GPU, change in `main.py`: `WhisperModel("turbo", device="cuda", compute_type="float16")` (and install CUDA-compatible dependencies).
- **Bot doesn’t respond** — Ensure **Message Content Intent** is enabled in the Developer Portal (Bot → Privileged Gateway Intents).

## Testing

Run the test suite (requires pytest; Discord and faster-whisper are mocked so no token or model is needed):

```bash
pytest tests/ -v
```

## Project structure

```
Watson/
├── main.py           # Bot, voice recording, faster-whisper transcription
├── tests/            # Basic tests (helpers, config, command mocks)
├── pytest.ini        # Pytest config
├── .env              # DISCORD_TOKEN, optional LOG_LEVEL, LOG_FILE (not committed)
├── .env.example      # Template for .env
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## License

Use and modify as you like.
