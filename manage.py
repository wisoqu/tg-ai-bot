"""Setup and operator CLI. Run `python manage.py` for the interactive menu."""
from __future__ import annotations

import asyncio
import getpass
import json
import os
import signal
import venv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def setup_environment() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        print(".env already exists; leaving it untouched. Edit it directly to change secrets.")
        return
    print("Create the minimum configuration. Secrets are entered without echo and saved only in the ignored .env file.")
    print("Choose an AI API: 1) Gemini API  2) Any OpenAI-compatible provider (OpenAI, OpenRouter, Groq, Together, local gateway, etc.)")
    provider_choice = input("Provider [1]: ").strip() or "1"
    if provider_choice not in {"1", "2"}:
        raise SystemExit("Choose 1 or 2.")
    telegram = getpass.getpass("Telegram bot token (from @BotFather): ").strip()
    while not telegram:
        print("A Telegram bot token is required.")
        telegram = getpass.getpass("Telegram bot token: ").strip()
    if provider_choice == "1":
        keys = getpass.getpass("Gemini API key(s), separated by spaces (from Google AI Studio): ").strip()
        while not keys:
            print("At least one Gemini API key is required.")
            keys = getpass.getpass("Gemini API key(s): ").strip()
        provider_config = f"AI_PROVIDER=gemini\nGEMINI_API_KEYS={keys}\nGEMINI_MODEL=auto\n"
        provider_config += "GEMINI_FALLBACK_MODELS=\nGEMINI_TRANSCRIPTION_MODEL=gemini-2.5-flash-lite\n"
    else:
        key = getpass.getpass("Provider API key: ").strip()
        while not key:
            print("An API key is required.")
            key = getpass.getpass("Provider API key: ").strip()
        base = input("OpenAI-compatible API base URL [https://api.openai.com/v1]: ").strip() or "https://api.openai.com/v1"
        model = input("Chat model ID (required): ").strip()
        while not model:
            model = input("Chat model ID (required): ").strip()
        transcription = input("Transcription model at /audio/transcriptions [whisper-1; blank disables voice]: ").strip()
        provider_config = f"AI_PROVIDER=openai-compatible\nAI_API_KEY={key}\nOPENAI_COMPAT_BASE_URL={base}\nOPENAI_COMPAT_MODEL={model}\nTRANSCRIPTION_MODEL={transcription}\n"
    text = (
        f"TELEGRAM_BOT_TOKEN={telegram}\n"
        + provider_config
        + "SQLITE_PATH=data/bot_memory.sqlite3\n"
        + "PRIVATE_DAILY_TOKEN_LIMIT=40000\n"
        + "GROUP_DAILY_TOKEN_LIMIT=150000\n"
        + "GLOBAL_DAILY_TOKEN_LIMIT=500000\n"
        + "MEMORY_RETENTION_DAYS=30\n"
    )
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as env_file:
        env_file.write(text)
    print("Created .env with owner-only permissions where supported. Next run: python manage.py check")


def load_config():
    from config import AppConfig
    return AppConfig.from_env()


def db_path(cfg) -> Path:
    path = Path(cfg.sqlite_path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def pid_path(cfg) -> Path:
    return Path(str(db_path(cfg)) + ".pid")

async def broadcast(cfg, store, text: str) -> None:
    from telegram import Bot
    bot = Bot(cfg.telegram_bot_token)
    async with bot:
        failed = 0
        for chat_id in store.known_chat_ids():
            try:
                await bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                failed += 1
                print(f"Could not notify chat {chat_id}: {exc}")
    print(f"Notification sent to {len(store.known_chat_ids()) - failed} chats; failed: {failed}.")

async def check_live(cfg) -> None:
    from telegram import Bot
    from ai_service import create_ai_service
    bot = Bot(cfg.telegram_bot_token)
    async with bot:
        identity = await bot.get_me()
        print(f"Telegram: OK — @{identity.username}")
    service = create_ai_service(cfg)
    if cfg.ai_provider == "gemini":
        models = await service._load_available_models()
        if not models:
            raise RuntimeError("Gemini model list is empty/unavailable. Check the API key, project access, and network.")
        print(f"Gemini: OK — {len(models)} models accessible; selected model: {cfg.ai_model}")
    else:
        print(f"AI API: OK — {await service.check_connection()}")
    close = getattr(service, "close", None)
    if close:
        await close()

async def run_command(command: str, cfg, store) -> None:
    if command == "check":
        path = db_path(cfg)
        store._init_db()
        print(f"Configuration: valid\nProvider: {cfg.ai_provider} / {cfg.ai_model}\nDatabase: {path}\nAPI keys configured: {len(cfg.gemini_api_keys) if cfg.ai_provider == 'gemini' else 1}\nStars packages: {len(cfg.star_token_packs)}\nDaily quotas: private {cfg.private_daily_token_limit:,}; group {cfg.group_daily_token_limit:,}; global {cfg.global_daily_token_limit:,}")
        if "--live" in sys.argv:
            await check_live(cfg)
        else:
            print("Use `python manage.py check --live` to verify Telegram and the selected AI provider over the network.")
    elif command == "status":
        path = db_path(cfg)
        size = path.stat().st_size if path.exists() else 0
        print(f"Service state: {'enabled' if store.get_setting('enabled', '1') == '1' else 'disabled'}")
        print(f"Known chats: {len(store.known_chat_ids())}\nSQLite: {path} ({size / 1024 / 1024:.2f} MiB / {cfg.max_database_mb} MiB target)")
        print(f"Running PID file: {pid_path(cfg)} ({'present' if pid_path(cfg).exists() else 'not found'})")
    elif command in ("disable", "enable"):
        enabled = command == "enable"
        default = "Бот временно недоступен. Мы сообщим, когда он снова заработает." if not enabled else "Бот снова доступен. Можно продолжать общение!"
        notice = input("Notification text (Enter for default): ").strip() or default
        store.set_setting("enabled", "1" if enabled else "0")
        await broadcast(cfg, store, notice)
        print("Bot enabled." if enabled else "Bot disabled.")
    elif command == "start":
        subprocess.run([sys.executable, str(ROOT / "main.py")], cwd=ROOT, check=False)
    elif command == "stop":
        notice = input("Notification before stop (Enter for default): ").strip() or "Бот будет временно недоступен."
        store.set_setting("enabled", "0")
        await broadcast(cfg, store, notice)
        path = pid_path(cfg)
        if not path.exists():
            print("No bot PID file found. If deployed with Docker/systemd, use that service manager to stop it.")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = int(data["pid"])
            if data.get("root") != str(ROOT):
                raise ValueError("PID file belongs to a different checkout")
            if sys.platform.startswith("linux"):
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if str(ROOT / "main.py") not in cmdline:
                    raise ValueError("PID is not running this bot; refusing to signal it")
            else:
                raise RuntimeError("Safe PID verification is only implemented on Linux; use the service manager.")
            os.kill(pid, signal.SIGTERM)
            print(f"Stop signal sent to bot PID {pid}.")
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"Safe stop failed: {exc}. Use your system service manager instead.")
    elif command == "vacuum":
        store.prune_expired(cfg.memory_retention_days)
        store.vacuum()
        print("Expired memory pruned and SQLite compacted.")
    elif command == "clear-memory":
        if input("This irreversibly clears ALL conversation history and saved facts. Type CLEAR: ").strip() == "CLEAR":
            store.clear_all_memory()
            store.vacuum()
            print("All chat memory was cleared. Purchases and token balances were preserved.")
        else:
            print("Cancelled.")
    elif command == "help":
        print("Commands: setup, install, check [--live], start, status, disable, enable, stop, vacuum, clear-memory")
    else:
        print(f"Unknown command: {command}. Use `python manage.py help`.")


def main() -> None:
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "menu"
    if command == "setup":
        setup_environment()
        return
    if command == "install":
        env_dir = ROOT / ".venv"
        if not (env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")).exists():
            venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
        print("Dependencies installed in .venv. Use ./botctl (or start_bot.ps1) to run the bot.")
        return
    if command == "menu" and not (ROOT / ".env").exists():
        setup_environment()
        return
    try:
        cfg = load_config()
        from storage import MemoryStore
        store = MemoryStore(str(db_path(cfg)))
    except (ValueError, OSError) as exc:
        print(f"Configuration error: {exc}\nRun `python manage.py setup` or edit .env from .env.example.", file=sys.stderr)
        raise SystemExit(2)
    if command == "menu":
        print("\nTG AI Bot\n1) Check setup\n2) Start bot (foreground)\n3) Status\n4) Disable + notify\n5) Enable + notify\n6) Stop + notify\n7) Prune/compact database\n8) Clear all chat memory\n0) Exit")
        option = input("> ").strip()
        command = {"1":"check", "2":"start", "3":"status", "4":"disable", "5":"enable", "6":"stop", "7":"vacuum", "8":"clear-memory", "0":"exit"}.get(option, "help")
    if command == "exit":
        return
    try:
        asyncio.run(run_command(command, cfg, store))
    except Exception as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
