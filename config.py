import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_SYSTEM_PROMPT = (
    "You are a practical and helpful assistant. "
    "Answer clearly, stay on task, and ask a clarifying question when needed."
)


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _env_models(name: str) -> list[str]:
    raw = _env_str(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    gemini_summary_model: str
    gemini_fallback_models: list[str]
    system_prompt: str
    sqlite_path: str
    trigger_mention: str
    max_context_tokens: int
    max_reply_tokens: int
    recent_messages: int
    group_recent_messages: int
    group_focus_messages: int
    retrieval_max_messages: int
    retrieval_lookback_messages: int
    retrieval_max_tokens: int
    summary_trigger_messages: int
    summary_input_tokens: int
    summary_output_tokens: int
    summary_max_chars: int
    temperature: float
    private_chat_limit_messages: int
    private_chat_limit_window_hours: int

    @staticmethod
    def from_env() -> "AppConfig":
        load_dotenv()

        telegram_bot_token = _env_str("TELEGRAM_BOT_TOKEN")
        gemini_api_key = _env_str("GEMINI_API_KEY")
        if not telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")

        primary_model = _env_str("GEMINI_MODEL", "auto")

        return AppConfig(
            telegram_bot_token=telegram_bot_token,
            gemini_api_key=gemini_api_key,
            gemini_model=primary_model,
            gemini_summary_model=_env_str("GEMINI_SUMMARY_MODEL", primary_model),
            gemini_fallback_models=_env_models("GEMINI_FALLBACK_MODELS"),
            system_prompt=_env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            sqlite_path=_env_str("SQLITE_PATH", "bot_memory.sqlite3"),
            trigger_mention=_env_str("TRIGGER_MENTION", "@ai"),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 2800),
            max_reply_tokens=_env_int("MAX_REPLY_TOKENS", 900),
            recent_messages=_env_int("RECENT_MESSAGES", 20),
            group_recent_messages=_env_int("GROUP_RECENT_MESSAGES", 28),
            group_focus_messages=_env_int("GROUP_FOCUS_MESSAGES", 18),
            retrieval_max_messages=_env_int("RETRIEVAL_MAX_MESSAGES", 10),
            retrieval_lookback_messages=_env_int("RETRIEVAL_LOOKBACK_MESSAGES", 320),
            retrieval_max_tokens=_env_int("RETRIEVAL_MAX_TOKENS", 520),
            summary_trigger_messages=_env_int("SUMMARY_TRIGGER_MESSAGES", 55),
            summary_input_tokens=_env_int("SUMMARY_INPUT_TOKENS", 2400),
            summary_output_tokens=_env_int("SUMMARY_OUTPUT_TOKENS", 280),
            summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 2200),
            temperature=_env_float("TEMPERATURE", 0.4),
            private_chat_limit_messages=_env_int("PRIVATE_CHAT_LIMIT_MESSAGES", 20),
            private_chat_limit_window_hours=_env_int("PRIVATE_CHAT_LIMIT_WINDOW_HOURS", 5),
        )
