"""Interactive/CLI operator console: python manage.py [status|disable|enable|stop]."""
import asyncio
import os
import signal
import sys
from pathlib import Path

from telegram import Bot

from config import AppConfig
from storage import MemoryStore


def paths(cfg):
    db = Path(cfg.sqlite_path)
    return db, Path(str(db) + ".pid")

async def broadcast(cfg, store, text):
    bot = Bot(cfg.telegram_bot_token)
    async with bot:
        for chat_id in store.known_chat_ids():
            try:
                await bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                print(f"Could not notify {chat_id}: {exc}")

async def run(command, cfg, store):
    if command == "status":
        print("Enabled:", store.get_setting("enabled", "1") == "1")
        print("Known chats:", len(store.known_chat_ids()))
        print("Database:", cfg.sqlite_path, f"({Path(cfg.sqlite_path).stat().st_size if Path(cfg.sqlite_path).exists() else 0} bytes)")
    elif command in ("disable", "enable"):
        state = command == "enable"
        default = "Бот временно недоступен. Мы сообщим, когда он снова заработает." if not state else "Бот снова доступен. Можно продолжать общение!"
        text = input(f"Текст уведомления (Enter — стандартный): ").strip() or default
        store.set_setting("enabled", "1" if state else "0")
        await broadcast(cfg, store, text)
        print("Bot enabled." if state else "Bot disabled.")
    elif command == "vacuum":
        store.prune_expired(cfg.memory_retention_days)
        store.vacuum()
        print("Expired memory removed and SQLite database compacted.")
    elif command == "clear-memory":
        confirm = input("This permanently clears ALL chats' history and saved facts. Type CLEAR to confirm: ")
        if confirm == "CLEAR":
            store.clear_all_memory()
            store.vacuum()
            print("All conversation and saved-fact memory cleared.")
        else:
            print("Cancelled.")
    elif command == "stop":
        text = input("Уведомление перед остановкой (Enter — стандартное): ").strip() or "Бот будет временно недоступен."
        store.set_setting("enabled", "0")
        await broadcast(cfg, store, text)
        _, pidfile = paths(cfg)
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                print(f"Stop signal sent to PID {pid}.")
            except (ValueError, ProcessLookupError, PermissionError) as exc:
                print(f"Could not stop process: {exc}")
        else:
            print("PID file not found; stop the bot through your service manager.")
    else:
        print("Commands: status, disable, enable, stop, vacuum, clear-memory")

def main():
    cfg = AppConfig.from_env()
    store = MemoryStore(cfg.sqlite_path)
    command = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if not command:
        print("TG AI Bot control\n1) Status\n2) Disable + notify\n3) Enable + notify\n4) Stop + notify\n5) Prune expired data + vacuum\n6) Clear all chat memory (irreversible)\n0) Exit")
        option = input("> ").strip()
        command = {"1":"status", "2":"disable", "3":"enable", "4":"stop", "5":"vacuum", "6":"clear-memory", "0":"exit"}.get(option, "help")
    if command == "exit": return
    asyncio.run(run(command, cfg, store))

if __name__ == "__main__": main()
