# Adding Watson Bot to Your Discord Server

This guide walks you through creating a Discord application, configuring the bot, and inviting it to your server.

---

## 1. Open the Discord Developer Portal

1. Go to **[discord.com/developers/applications](https://discord.com/developers/applications)**.
2. Log in with your Discord account if prompted.

---

## 2. Create a New Application

1. Click **"New Application"**.
2. Enter a name (e.g. **Watson**), accept the terms, and click **"Create"**.
3. You are now on the application’s **General Information** page. You can set an icon and description later if you want.

---

## 3. Create the Bot User

1. In the left sidebar, open **"Bot"**.
2. Click **"Add Bot"** and confirm.
3. Optionally set a **username** and **avatar** for the bot (same page or via "General Information").

---

## 4. Enable Required Privileged Intents

Watson needs two **Privileged Gateway Intents** to read messages and see members.

1. Still in **Bot**, scroll to **"Privileged Gateway Intents"**.
2. Turn **ON**:
   - **Presence Intent** — optional (leave off if you don’t need it).
   - **Server Members Intent** — **required** (bot needs to see members).
   - **Message Content Intent** — **required** (bot needs to read `!commands`).
3. Click **"Save Changes"**.

---

## 5. Copy the Bot Token

1. On the **Bot** page, under **"TOKEN"**, click **"Reset Token"** (or **"View Token"** if you already have one).
2. Confirm and **copy the token**.
3. Put it in your project’s `.env` file:
   ```env
   DISCORD_TOKEN=paste_your_token_here
   ```
4. **Never commit the token to git or share it.** If it’s leaked, use **"Reset Token"** in the portal and update `.env`.

---

## 6. Set the Invite URL (OAuth2)

1. In the left sidebar, open **"OAuth2"** → **"URL Generator"**.
2. **SCOPES:** enable:
   - **bot** — to add the bot to the server.
   - **applications.commands** — optional; only needed if you add slash commands later.
3. **BOT PERMISSIONS:** enable at least:
   - **View Channels** — see text and voice channels.
   - **Connect** — join voice channels.
   - **Speak** — required for voice (recording).
   - **Send Messages** — send transcript and replies.
   - **Read Message History** — read commands and context.
   - **Attach Files** — send `transcript.txt` / `analysis.txt` when messages are long.
4. The page will show a **"Generated URL"** at the bottom. Copy it.

---

## 7. Invite the Bot to Your Server

1. Paste the **Generated URL** into your browser and press Enter.
2. In **"Add to Server"**, choose the server you want Watson in (you must have **"Manage Server"** or **"Administrator"** there).
3. Click **"Continue"**, review the permissions, then click **"Authorize"**.
4. Complete the captcha if asked.
5. The bot will appear in your server’s member list (offline until you run `main.py`).

---

## 8. Run the Bot

1. In the project folder, activate the virtual environment and run:
   ```bash
   python main.py
   ```
2. When you see something like `Watson Online` and the list of guilds, the bot is connected.
3. In any **text channel** where the bot can read messages, type **`!join`** (while you are in a voice channel), then **`!record`** to start recording. Use **`!stop`** to get the transcript.

---

## Quick Checklist

| Step | What to do |
|------|------------|
| 1 | Create application at [discord.com/developers/applications](https://discord.com/developers/applications) |
| 2 | Bot → Add Bot |
| 3 | Bot → Enable **Server Members Intent** and **Message Content Intent** |
| 4 | Copy token → put in `.env` as `DISCORD_TOKEN=...` |
| 5 | OAuth2 → URL Generator → scopes: **bot** → permissions: View Channels, Connect, Speak, Send Messages, Read Message History, Attach Files |
| 6 | Open Generated URL → choose server → Authorize |
| 7 | Run `python main.py` and use `!join` / `!record` / `!stop` in Discord |

---

## Troubleshooting

- **Bot doesn’t respond to `!check` / `!record`**  
  Ensure **Message Content Intent** is enabled under Bot → Privileged Gateway Intents, then restart `main.py`.

- **Bot can’t see members or voice**  
  Enable **Server Members Intent** and ensure the invite URL included **View Channels**, **Connect**, and **Speak**.

- **"Missing Access" or no permission to add bot**  
  You need **Manage Server** or **Administrator** on the target server. Ask an admin to use the invite URL or to grant you that role.

- **Token invalid / 401**  
  Reset the token in the Developer Portal (Bot → Reset Token) and update `DISCORD_TOKEN` in `.env`.
