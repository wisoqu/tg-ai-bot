# Telegram Gemini Bot (Groups + Private Chat)

Telegram bot with rolling chat memory in SQLite and response generation via Google Gemini.

## Behavior

- Private chat: responds to normal dialogue.
- Private chat limit: `20 messages / 5 hours` per user.
- Group chat: responds when:
  - bot username is mentioned (`@your_bot`)
  - custom trigger appears (default `@ai`)
  - user replies to bot's message
- Group chats: no per-user limit.

## Memory and Context

- SQLite storage (`bot_memory.sqlite3`)
- Hybrid memory:
  - recent dialogue window
  - rolling summary
  - relevant older messages retrieved by query
- Group optimization:
  - larger recent window for groups
  - focus boost for current speaker + bot messages
- Configurable context/output token caps for model requests

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and fill:
   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY`
   - keep `GEMINI_MODEL=auto` for automatic model selection

3. Run:

```bash
python main.py
```

or on PowerShell:

```powershell
.\start_bot.ps1
```

## Commands

- `/start`
- `/status`
- `/reset`

## Notes

- This project uses Gemini native SDK (`google-genai`), not an OpenAI-compatible endpoint.
- Auto mode tries multiple Gemini models and falls back when one model is rate-limited.
- If Gemini returns quota errors, check billing and quota limits in your Google AI project.
