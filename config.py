import os
import re
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
    return [item.strip() for item in re.split(r"[\s,]+", raw) if item.strip()]


def _parse_star_packs(raw: str) -> list[tuple[int, int]]:
    packs: list[tuple[int, int]] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError("STAR_TOKEN_PACKS entries must use TOKENS:STARS")
        try:
            tokens, stars = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("STAR_TOKEN_PACKS values must be integers") from exc
        if tokens <= 0 or stars <= 0:
            raise ValueError("STAR_TOKEN_PACKS tokens and Stars must be positive")
        packs.append((tokens, stars))
    if len(packs) > 10:
        raise ValueError("Configure at most 10 STAR_TOKEN_PACKS")
    return packs


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    ai_provider: str
    ai_api_key: str
    ai_model: str
    openai_compat_base_url: str
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
    temperature: float
    gemini_api_keys: list[str]
    private_daily_token_limit: int
    group_daily_token_limit: int
    global_daily_token_limit: int
    memory_retention_days: int
    max_voice_seconds: int
    transcription_model: str
    max_database_mb: int
    star_token_packs: list[tuple[int, int]]

    @staticmethod
    def from_env() -> "AppConfig":
        load_dotenv()

        telegram_bot_token = _env_str("TELEGRAM_BOT_TOKEN")
        provider = _env_str("AI_PROVIDER", "gemini").lower()
        if provider not in {"gemini", "openai-compatible"}:
            raise ValueError("AI_PROVIDER must be 'gemini' or 'openai-compatible'")
        configured_keys = [key for key in _env_models("GEMINI_API_KEYS") if not key.startswith("your_")]
        legacy_key = _env_str("GEMINI_API_KEY")
        if legacy_key.startswith("your_"):
            legacy_key = ""
        generic_key = _env_str("AI_API_KEY") or _env_str("OPENAI_COMPAT_API_KEY")
        if provider == "gemini":
            gemini_keys = list(dict.fromkeys(([legacy_key] if legacy_key else []) + configured_keys))
            provider_key = gemini_keys[0] if gemini_keys else ""
        else:
            gemini_keys = []
            provider_key = generic_key
        if not telegram_bot_token or telegram_bot_token == "your_telegram_token_here":
            raise ValueError("Set TELEGRAM_BOT_TOKEN in .env")
        if not provider_key or provider_key.startswith("your_"):
            if provider == "gemini":
                raise ValueError("Set GEMINI_API_KEY or GEMINI_API_KEYS in .env")
            raise ValueError("Set AI_API_KEY in .env")

        primary_model = _env_str("GEMINI_MODEL", "auto") if provider == "gemini" else _env_str("OPENAI_COMPAT_MODEL", "")
        base_url = _env_str("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if provider == "openai-compatible":
            if not primary_model:
                raise ValueError("Set OPENAI_COMPAT_MODEL when AI_PROVIDER=openai-compatible")
            if not base_url.startswith(("https://", "http://")):
                raise ValueError("OPENAI_COMPAT_BASE_URL must be an http(s) URL")

        cfg = AppConfig(
            telegram_bot_token=telegram_bot_token,
            ai_provider=provider,
            ai_api_key=provider_key,
            ai_model=primary_model,
            openai_compat_base_url=base_url,
            gemini_fallback_models=_env_models("GEMINI_FALLBACK_MODELS"),
            system_prompt=_env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            sqlite_path=_env_str("SQLITE_PATH", "bot_memory.sqlite3"),
            trigger_mention=_env_str("TRIGGER_MENTION", "@ai"),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 2800),
            max_reply_tokens=_env_int("MAX_REPLY_TOKENS", 900),
            recent_messages=_env_int("RECENT_MESSAGES", 20),
            group_recent_messages=_env_int("GROUP_RECENT_MESSAGES", 20),
            group_focus_messages=_env_int("GROUP_FOCUS_MESSAGES", 0),
            retrieval_max_messages=_env_int("RETRIEVAL_MAX_MESSAGES", 10),
            retrieval_lookback_messages=_env_int("RETRIEVAL_LOOKBACK_MESSAGES", 320),
            retrieval_max_tokens=_env_int("RETRIEVAL_MAX_TOKENS", 520),
            temperature=_env_float("TEMPERATURE", 0.4),
            gemini_api_keys=gemini_keys,
            private_daily_token_limit=_env_int("PRIVATE_DAILY_TOKEN_LIMIT", 40000),
            group_daily_token_limit=_env_int("GROUP_DAILY_TOKEN_LIMIT", 150000),
            global_daily_token_limit=_env_int("GLOBAL_DAILY_TOKEN_LIMIT", 500000),
            memory_retention_days=_env_int("MEMORY_RETENTION_DAYS", 30),
            max_voice_seconds=_env_int("MAX_VOICE_SECONDS", 120),
            transcription_model=(
                _env_str("GEMINI_TRANSCRIPTION_MODEL", "gemini-2.5-flash-lite")
                if provider == "gemini"
                else _env_str("TRANSCRIPTION_MODEL", "whisper-1")
            ),
            max_database_mb=_env_int("MAX_DATABASE_MB", 500),
            star_token_packs=_parse_star_packs(_env_str("STAR_TOKEN_PACKS", "")),
        )
        positive_values = {
            "MAX_CONTEXT_TOKENS": cfg.max_context_tokens,
            "MAX_REPLY_TOKENS": cfg.max_reply_tokens,
            "PRIVATE_DAILY_TOKEN_LIMIT": cfg.private_daily_token_limit,
            "GROUP_DAILY_TOKEN_LIMIT": cfg.group_daily_token_limit,
            "GLOBAL_DAILY_TOKEN_LIMIT": cfg.global_daily_token_limit,
            "MEMORY_RETENTION_DAYS": cfg.memory_retention_days,
            "MAX_VOICE_SECONDS": cfg.max_voice_seconds,
            "MAX_DATABASE_MB": cfg.max_database_mb,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Settings must be positive integers: {', '.join(invalid)}")
        if not 0 <= cfg.temperature <= 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        return cfg
