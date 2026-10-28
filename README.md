# Telegram AI Bot — open-source starter kit

A self-hosted Telegram assistant for private chats and groups. Bring your own Telegram bot token and an AI API key; the project does not contain or provide credentials. Gemini and OpenAI-compatible providers (OpenAI, OpenRouter, Groq, Together, compatible gateways) are supported. Text, recent group context, explicit memory, addressed voice transcription, optional Telegram Stars packages, and operator tools are included.

- **Python:** 3.10+
- **AI backends:** Google Gemini Developer API or OpenAI-compatible Chat Completions API
- **Storage:** SQLite, no external database required
- **License:** MIT — see [`LICENSE`](LICENSE)

[Русская инструкция](README_RU.md) · [Security notes](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## Fastest setup (Docker Compose)

1. Install Docker Engine + Compose plugin.
2. Create a Telegram bot with [@BotFather](https://t.me/BotFather) and copy its token.
3. Choose any supported provider. For Gemini, create a key in [Google AI Studio](https://aistudio.google.com/apikey). For other providers, copy their API key, base URL, and model ID from their documentation.
4. In the repository folder:

   ```sh
   cp .env.example .env
   ```

   Set `TELEGRAM_BOT_TOKEN` and select your provider in `.env`. For Gemini, set `AI_PROVIDER=gemini` and `GEMINI_API_KEY` (or `GEMINI_API_KEYS`). For OpenAI-compatible APIs, set `AI_PROVIDER=openai-compatible`, `AI_API_KEY`, `OPENAI_COMPAT_BASE_URL`, and `OPENAI_COMPAT_MODEL`. Keep this file private; it is git-ignored.
5. Start the bot:

   ```sh
   docker compose up -d --build
   docker compose logs -f bot
   ```

   Conversation data persists in a Docker named volume (`tg-ai-bot_bot_data` by default). Stop with `docker compose down` (volume stays); upgrade with `git pull && docker compose up -d --build`. Do not add `-v` to `down` unless you intend to erase the database. Run operator commands inside the same container so they use the same SQLite volume, e.g. `docker compose exec bot python manage.py status`; for interactive actions add `-it`.

No provider key is shipped with this project. Gemini auto mode intentionally prefers economical Flash-Lite models. OpenAI-compatible mode accepts a provider-specific key, endpoint, and model ID. Explicit Gemini fallback models are opt-in: set `GEMINI_FALLBACK_MODELS` only after checking their availability and pricing. Providers may change model IDs, free access, quotas, and API compatibility.

## Setup on Windows, macOS, or Linux (without Docker)

1. Install Python 3.10+ and Git.
2. Clone the repository and enter its folder.
3. Run:

   ```sh
   python manage.py setup
   ```

   The wizard asks which AI provider you want, then collects the Telegram token and provider credentials without echoing secrets, and creates a private `.env` file. On Windows, `start_bot.ps1` creates a virtual environment, installs requirements, runs the wizard if needed, validates configuration, and launches the bot. On macOS/Linux, run `./botctl` for the operator menu (or `python3 manage.py`).
4. Run the bot. On **Windows**, use PowerShell:

   ```powershell
   .\start_bot.ps1
   ```

   The script creates `.venv`, installs dependencies, runs the wizard if needed, checks local configuration, then starts the bot. To optionally test the Telegram and AI credentials over the network after setup, run `.\.venv\Scripts\python.exe manage.py check --live` in another PowerShell window. On **macOS/Linux**, validate and launch with:

   ```sh
   ./botctl check --live
   ./botctl start
   ```

   The bot runs in the foreground; use `Ctrl+C` to stop. For unattended production hosting, prefer Docker Compose or a process manager such as systemd. A starter unit is in [`deploy/tg-ai-bot.service.example`](deploy/tg-ai-bot.service.example); replace its paths and service account for your host before enabling it.

### Telegram group setup

Add the bot to the group. If it should use the surrounding conversation as context, open BotFather → `/setprivacy` → select the bot → **Disable**, then remove and re-add it to the group if Telegram asks. This means Telegram will deliver ordinary group messages to the bot. Make this behavior clear to group members and get any consent required for your deployment. If privacy mode remains on, the bot only receives the updates Telegram permits (such as commands, mentions, and replies), so the context will be incomplete. The bot responds to explicit mentions, the configured `TRIGGER_MENTION` (default `@ai`), or a reply to the bot. Group voice notes are transcribed only when addressed by mention/reply.

## What is configurable

All settings are in `.env`; `.env.example` documents defaults.

| Setting | Default | Purpose |
|---|---:|---|
| `AI_PROVIDER` | `gemini` | `gemini` or `openai-compatible` |
| `AI_API_KEY` | empty | Key for an OpenAI-compatible endpoint |
| `OPENAI_COMPAT_BASE_URL` | OpenAI API URL | OpenAI-compatible API root (e.g. provider `/v1`) |
| `OPENAI_COMPAT_MODEL` | empty | Model ID required for compatible providers |
| `GEMINI_MODEL` | `auto` | Economical Flash-Lite preference list; can be set to an explicit model ID |
| `GEMINI_API_KEYS` | empty | Optional Gemini keys separated by spaces or commas; `GEMINI_API_KEY` also works |
| `PRIVATE_DAILY_TOKEN_LIMIT` | 40,000 | Free daily app quota per private user |
| `GROUP_DAILY_TOKEN_LIMIT` | 150,000 | Shared daily app quota per group; no group top-ups |
| `GLOBAL_DAILY_TOKEN_LIMIT` | 500,000 | Per-bot daily safety ceiling across chats |
| `STAR_TOKEN_PACKS` | empty | Optional `TOKENS:STARS` packages, e.g. `100000:50,500000:200` |
| `MEMORY_RETENTION_DAYS` | 30 | Rolling retention for conversation messages and explicit facts |
| `MAX_DATABASE_MB` | 500 | Database size at which the bot runs a compaction pass |
| `MAX_VOICE_SECONDS` | 120 | Maximum voice-note duration |
| `GEMINI_TRANSCRIPTION_MODEL` / `TRANSCRIPTION_MODEL` | Provider-dependent | Voice transcription model; compatible APIs must implement `/audio/transcriptions` |
| `SQLITE_PATH` | `bot_memory.sqlite3` | SQLite file location |

Daily token quotas are **application limits**, not promises about Google's free allowance or API billing. Exact Gemini quotas depend on the project/model and are shown in AI Studio. API keys in the same Google project share the same provider quota; rotating keys from one project does not multiply it. Multiple keys are useful for credential failover and, only when independently configured, project-level fallback. Compatible-provider limits and pricing are controlled by that provider. The global limit is a further guardrail, not a billing guarantee.

When available, actual input/output usage metadata is recorded; otherwise the bot estimates token counts from text. `/usage` shows approximate words and typical exchanges—the actual context length affects each request. Daily counters reset at 00:00 UTC.

### Optional Telegram Stars

Stars payments are disabled until the operator deliberately sets `STAR_TOKEN_PACKS`. Example format: `STAR_TOKEN_PACKS=100000:50,500000:200` means tokens then Stars. Choose prices yourself, ensure the bot's terms/refund/support practices are appropriate for your jurisdiction, and test with a Telegram test bot before real users. `/buy` lists packages; `/buy 1` sends only the selected invoice. Packages top up a private user's balance; group quotas cannot be topped up. Telegram payment confirmation is checked before a purchase is credited.

## Bot commands

- `/start` — onboarding and group-use instructions
- `/status` — memory and model settings
- `/usage` — remaining daily app quota, estimate, and private paid balance
- `/buy` — list configured Stars packages; `/buy 1` to purchase a listed package
- `/remember <fact>` — explicitly save a fact for the retention period
- `/forget` — clear saved facts (group admins only in groups)
- `/reset` — clear chat history and saved facts (group admins only in groups)

The bot stores ordinary text messages it receives in groups to build context. In a personal chat, `/reset` clears that chat. Group administrator checks require the bot to be able to query group membership; make it an admin if Telegram denies those checks.

## Operator console

```sh
./botctl                 # menu (macOS/Linux)
python manage.py         # menu (cross-platform)
python manage.py check   # local config + DB check
python manage.py check --live  # test Telegram and the selected AI provider
python manage.py status
python manage.py disable # asks for announcement, pauses responses
python manage.py enable  # asks for announcement, resumes responses
python manage.py vacuum # purge expired data and compact SQLite
python manage.py clear-memory # destructive, asks for confirmation
```

`python manage.py stop` broadcasts a notice and safely signals the recorded bot process on Linux. When deployed with Docker/systemd, stop it with `docker compose down` or `systemctl stop <service>` so the supervisor does not restart it. The bot automatically prunes expired records hourly while running; SQLite is compacted when it reaches `MAX_DATABASE_MB` and can also be compacted manually. With Compose, run e.g. `docker compose exec -it bot python manage.py vacuum`.

## Privacy, safety, and operations

- The operator supplies and controls all API keys and pays any provider charges. Free-tier access and model availability can change; check provider settings before exposing the bot publicly.
- The selected AI provider receives prompts built from the current message and selected chat context. Addressed voice notes are sent to Gemini or the configured compatible transcription endpoint. Provider data handling depends on your provider/account tier and current terms.
- Group messages and explicit facts are stored locally in SQLite for the configured retention period. Use `/reset` to clear the chat; group members should be told when context capture is enabled.
- Never commit `.env`, API keys, Telegram tokens, or the SQLite database. Back up `data/` securely if needed.
- This project is a starter kit, not a hosted service or uptime guarantee. Set Telegram/Gemini billing alerts and project quotas yourself.

## Development

```sh
python -m venv .venv
# activate the environment, then:
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Bug reports should include Python version, OS/deployment mode, and sanitized logs—not tokens or private chat content.
