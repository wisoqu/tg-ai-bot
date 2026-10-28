# Telegram Gemini Bot

Telegram assistant for private chats and groups, powered by Gemini and backed by SQLite.

## Features

- Private chats: responds to text and voice messages.
- Groups: responds to bot mentions, `@ai`, replies to the bot, and voice messages addressed by mention/reply. Ordinary text received from Telegram is stored to provide up to about 20 recent messages as group context. To receive ordinary group messages, disable BotFather privacy mode (`/setprivacy` → Disable) and ensure the bot has permission to read them.
- Daily application token quotas: 40,000 tokens per private user and 150,000 tokens per group by default; configure with `PRIVATE_DAILY_TOKEN_LIMIT` / `GROUP_DAILY_TOKEN_LIMIT`. Usage charges actual Gemini usage metadata where available, otherwise an estimate. `/usage` reports remaining tokens plus rough words and average exchanges. These are bot policy limits, not Google's free-tier quotas.
- Explicit saved facts: `/remember text` saves a fact for 30 days; `/forget` removes saved facts. In groups, only admins may use these commands.
- Telegram Stars payments: set `STAR_TOKEN_PACKS` as comma-separated `TOKENS:STARS` pairs to enable `/buy` in private chat. Payments are verified against the invoice and credited idempotently; group quotas cannot be topped up. No package prices are enabled by default until you select them.
- Message history and saved facts are stored in SQLite. Messages and saved facts expire after 30 days (`MEMORY_RETENTION_DAYS`). Old rolling summaries are disabled because their source data cannot reliably be expired independently.
- Voice notes are downloaded and sent for transcription only in private chats or when addressed to the bot in a group. The transcription is then sent to the text model. Audio processing itself consumes Gemini quota; maximum duration defaults to 120 seconds.
- Multiple Gemini keys can be supplied via `GEMINI_API_KEYS`, separated by spaces or commas, with rotation on credential/quota errors. Google's limits are per project, not per key; multiple keys in one project do not multiply quota.
- Interactive operator console: `python manage.py` (or `python manage.py status|disable|enable|stop|vacuum|clear-memory`). Disable/enable broadcasts a custom or default notice to chats the bot has seen. Stop sends a notice and signals the bot process; if managed by systemd/Docker, use that service manager for restart. `vacuum` prunes expired data and compacts SQLite; `clear-memory` requires explicit confirmation and irreversibly clears all chats' message/fact memory.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `TELEGRAM_BOT_TOKEN` and either `GEMINI_API_KEY` or `GEMINI_API_KEYS` (keys separated by spaces). Keep `.env` private.
3. Run `python main.py`.

For a server deployment, run the bot under `systemd` or Docker and use `python manage.py` from the project directory. `manage.py` reads the same environment and SQLite path as the bot.

## Telegram commands

- `/start` — usage instructions
- `/status` — memory/model status
- `/usage` — daily token use, remaining quota, and paid balance
- `/buy` — purchase a configured Stars package in private chat
- `/remember text` — explicitly save a fact
- `/forget` — clear saved facts (group admins only in groups)
- `/reset` — clear chat history and saved facts

## Gemini free tier

Google does not publish a universal fixed number of free tokens per API key. Gemini API requests are constrained by model/project-specific RPM, TPM and RPD limits shown in Google AI Studio. All API keys belonging to a project share that project's quota. Check the AI Studio rate-limit page for the active limits on your project.
