# Watson

A Discord bot that records voice channel audio, transcribes it with [OpenAI Whisper](https://github.com/openai/whisper), and optionally summarizes the conversation using a local [Ollama](https://ollama.ai) model.

## Features

- **Voice recording** — Joins a voice channel and records participants
- **Transcription** — Converts speech to text via Whisper (turbo model)
- **Transcript** — Posts a timestamped transcript in the channel (or as a file if long)
- **Optional AI summary** — Sends the transcript to Ollama for a brief summary and action items (if Ollama is running)

## Prerequisites

- Python 3.10+
- [Discord Bot Token](https://discord.com/developers/applications) — create an application and bot, enable **Message Content Intent** and **Server Members Intent**
- **macOS (Homebrew):** Opus is loaded from `/opt/homebrew/lib/libopus.dylib`; install with `brew install opus` if needed

## Setup

1. **Clone and create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies: `py-cord`, `openai-whisper`, `python-dotenv`, `requests`, `torch`.

3. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your Discord bot token:

   ```
   DISCORD_TOKEN=your_bot_token_here
   ```

4. **Optional — Ollama (for AI summary)**

   If you want the “analysis & conclusions” step:

   - Install [Ollama](https://ollama.ai) and run: `ollama run llama3`
   - The bot will call `http://localhost:11434/api/generate`. If Ollama is not running, the bot will still post the transcript and report that analysis failed.

## Usage

1. Invite the bot to your server (OAuth2 URL with scopes: `bot`, `applications.commands`; permissions: **Connect**, **Speak**, **Use Voice Activity**, **Send Messages**, **Read Message History**).
2. In a text channel, run:

   | Command   | Description                          |
   | --------- | ------------------------------------ |
   | `!join`   | Bot joins your current voice channel |
   | `!record` | Start recording and transcription   |
   | `!stop`   | Stop recording and get the transcript |
   | `!leave`  | Bot leaves the voice channel         |
   | `!check`  | Test connection and bot permissions  |

3. After `!stop`, the bot posts the transcript and, if Ollama is available, an analysis/summary.

## Project structure

```
Watson/
├── main.py           # Bot logic, Whisper transcription, Ollama integration
├── .env              # DISCORD_TOKEN (not committed)
├── .env.example      # Template for .env
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## License

Use and modify as you like.
