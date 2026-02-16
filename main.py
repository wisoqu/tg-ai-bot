import asyncio
import html
import logging
import re
import time
from typing import Any

from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    ChatMemberUpdated,
    Message,
    MessageEntity,
    Update,
)
from telegram.constants import ChatAction, ChatMemberStatus, ParseMode
from telegram.ext import (
    Application,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import AppConfig
from gemini_service import GeminiQuotaError, GeminiService
from storage import MemoryStore


LOGGER = logging.getLogger(__name__)
MAX_TELEGRAM_MESSAGE_LEN = 4000
MAX_TELEGRAM_CODE_BLOCK_LEN = 3600
CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n?(.*?)```", re.DOTALL)
CODE_STYLE_INSTRUCTION = (
    "When you provide code, wrap it in fenced code blocks with a language tag. "
    "Example: ```python\\nprint('hello')\\n```."
)
MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
MD_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*([^*\n][^*]*?)\*(?!\*)", re.DOTALL)
MD_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)_([^\n_]+?)_(?!\w)", re.DOTALL)
MD_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
MD_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def is_group_chat(chat_type: str | None) -> bool:
    return chat_type in {"group", "supergroup"}


def message_mentions_bot_username(message: Message, bot_username: str) -> bool:
    if not bot_username:
        return False
    target = f"@{bot_username.lower()}"
    for mention_text in message.parse_entities(types=[MessageEntity.MENTION]).values():
        if mention_text.lower() == target:
            return True
    return False


def contains_trigger_mention(text: str, trigger_mention: str) -> bool:
    if not trigger_mention:
        return False
    pattern = rf"(^|\s){re.escape(trigger_mention)}\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def should_respond(
    message: Message,
    bot_username: str,
    trigger_mention: str,
    bot_id: int | None,
) -> bool:
    if not message.text:
        return False

    if not is_group_chat(message.chat.type):
        return True

    if message.reply_to_message and message.reply_to_message.from_user and bot_id:
        if message.reply_to_message.from_user.id == bot_id:
            return True

    if message_mentions_bot_username(message, bot_username):
        return True

    if contains_trigger_mention(message.text, trigger_mention):
        return True

    return False


def sanitize_user_text(text: str, bot_username: str, trigger_mention: str) -> str:
    cleaned = text
    if bot_username:
        cleaned = re.sub(
            rf"@{re.escape(bot_username)}\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    if trigger_mention:
        cleaned = re.sub(
            rf"{re.escape(trigger_mention)}\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return re.sub(r"\s+", " ", cleaned).strip(" \t\n,:-")


def format_wait_time(seconds: int) -> str:
    if seconds <= 0:
        return "a few minutes"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h"
    return f"{max(1, minutes)}m"


def format_user_text(message: Message, clean_text: str) -> str:
    if is_group_chat(message.chat.type):
        sender_name = message.from_user.full_name if message.from_user else "User"
        return f"{sender_name}: {clean_text}"
    return clean_text


def build_generation_messages(
    cfg: AppConfig,
    summary: str,
    history: list[dict[str, Any]],
    user_text: str,
    relevant_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    used_tokens = approx_tokens(cfg.system_prompt)
    messages: list[dict[str, str]] = []

    if summary:
        summary_message = f"Conversation summary:\n{summary}"
        messages.append({"role": "system", "content": summary_message})
        used_tokens += approx_tokens(summary_message)

    relevant_message = _build_relevant_memory_message(cfg, relevant_records or [])
    if relevant_message:
        messages.append({"role": "system", "content": relevant_message})
        used_tokens += approx_tokens(relevant_message)

    selected_history: list[dict[str, str]] = []
    for item in reversed(history):
        turn_cost = approx_tokens(item["content"]) + 6
        if used_tokens + turn_cost > cfg.max_context_tokens:
            break
        selected_history.append(item)
        used_tokens += turn_cost

    messages.extend(reversed(selected_history))
    messages.append({"role": "user", "content": user_text})
    return messages


def _build_relevant_memory_message(cfg: AppConfig, records: list[dict[str, Any]]) -> str:
    if not records:
        return ""

    lines: list[str] = []
    consumed_tokens = 0
    for item in records:
        line = f"{item['role']}: {item['content']}"
        line_tokens = approx_tokens(line) + 4
        if consumed_tokens + line_tokens > cfg.retrieval_max_tokens:
            break
        lines.append(f"- {line}")
        consumed_tokens += line_tokens

    if not lines:
        return ""
    return "Relevant older context:\n" + "\n".join(lines)


def collect_context_records(
    *,
    cfg: AppConfig,
    store: MemoryStore,
    chat_id: int,
    user_text: str,
    is_group: bool,
    current_user_id: int | None,
    bot_id: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recent_limit = cfg.group_recent_messages if is_group else cfg.recent_messages
    recent_records = store.fetch_recent_records(chat_id, recent_limit)
    selected: dict[int, dict[str, Any]] = {int(item["id"]): item for item in recent_records}

    if is_group:
        focus_ids: list[int] = []
        if current_user_id is not None:
            focus_ids.append(current_user_id)
        if bot_id is not None:
            focus_ids.append(bot_id)
        for item in store.fetch_recent_by_authors(chat_id, focus_ids, cfg.group_focus_messages):
            selected.setdefault(int(item["id"]), item)

    selected_records = sorted(selected.values(), key=lambda item: int(item["id"]))
    relevant_records = store.fetch_relevant_records(
        chat_id=chat_id,
        query=user_text,
        limit=cfg.retrieval_max_messages,
        lookback_messages=cfg.retrieval_lookback_messages,
        prefer_author_id=current_user_id if is_group else None,
        bot_author_id=bot_id,
        exclude_ids={int(item["id"]) for item in selected_records},
    )
    return selected_records, relevant_records


def split_for_telegram(text: str, max_len: int = MAX_TELEGRAM_MESSAGE_LEN) -> list[str]:
    body = (text or "").strip()
    if not body:
        return [""]

    chunks: list[str] = []
    rest = body
    while len(rest) > max_len:
        cut = rest.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = rest.rfind(" ", 0, max_len)
        if cut <= 0:
            cut = max_len
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    chunks.append(rest)
    return chunks


def markdownish_text_to_html(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        header_level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= header_level <= 6 and stripped.startswith("#" * header_level + " "):
            header_content = stripped[header_level + 1 :].strip()
            lines.append(f"<b>{html.escape(header_content)}</b>")
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            lines.append(f"{html.escape(indent)}• {html.escape(stripped[2:])}")
            continue

        lines.append(html.escape(line))

    payload = "\n".join(lines)
    payload = MD_LINK_RE.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        payload,
    )
    payload = MD_INLINE_CODE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", payload)
    payload = MD_BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", payload)
    payload = MD_STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", payload)
    payload = MD_ITALIC_STAR_RE.sub(lambda m: f"<i>{m.group(1)}</i>", payload)
    payload = MD_ITALIC_UNDERSCORE_RE.sub(lambda m: f"<i>{m.group(1)}</i>", payload)
    return payload


def split_code_for_telegram(code: str, max_len: int = MAX_TELEGRAM_CODE_BLOCK_LEN) -> list[str]:
    content = code.rstrip()
    if not content:
        return [""]
    if len(content) <= max_len:
        return [content]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in content.splitlines(keepends=True):
        line_len = len(line)
        if line_len > max_len:
            if current:
                chunks.append("".join(current).rstrip("\n"))
                current = []
                current_len = 0
            rest = line
            while len(rest) > max_len:
                chunks.append(rest[:max_len].rstrip("\n"))
                rest = rest[max_len:]
            if rest:
                current = [rest]
                current_len = len(rest)
            continue

        if current_len + line_len > max_len and current:
            chunks.append("".join(current).rstrip("\n"))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("".join(current).rstrip("\n"))

    return chunks or [content]


def looks_like_python_code(text: str) -> bool:
    sample = text.strip()
    if sample.count("\n") < 2:
        return False
    markers = [
        "def ",
        "class ",
        "import ",
        "from ",
        "print(",
        "if __name__",
        "return ",
        "for ",
        "while ",
        "try:",
        "except ",
    ]
    score = sum(1 for marker in markers if marker in sample)
    return score >= 2


def extract_response_segments(text: str) -> list[tuple[str, str, str]]:
    segments: list[tuple[str, str, str]] = []
    cursor = 0

    for match in CODE_FENCE_RE.finditer(text):
        start, end = match.span()
        if start > cursor:
            plain = text[cursor:start]
            if plain.strip():
                segments.append(("text", "", plain.strip()))

        lang = (match.group(1) or "").strip().lower()
        code = (match.group(2) or "").strip("\n")
        if code.strip():
            segments.append(("code", lang, code))
        cursor = end

    tail = text[cursor:]
    if tail.strip():
        segments.append(("text", "", tail.strip()))

    if segments:
        return segments

    raw = text.strip()
    if looks_like_python_code(raw):
        return [("code", "python", raw)]
    return [("text", "", raw)]


class TelegramGeminiBot:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.store = MemoryStore(cfg.sqlite_path)
        self.llm = GeminiService(
            api_key=cfg.gemini_api_key,
            fallback_models=cfg.gemini_fallback_models,
        )
        self.chat_locks: dict[int, asyncio.Lock] = {}

    def _build_usage_text(self, bot_username: str = "") -> str:
        username_hint = f"@{bot_username}" if bot_username else "@bot_username"
        return (
            "Hi. I am your Gemini chat bot.\n"
            f"Group trigger: {self.cfg.trigger_mention}\n"
            f"In groups I respond if you mention {username_hint}, use "
            f"{self.cfg.trigger_mention}, or reply to my message.\n"
            f"Private limit: {self.cfg.private_chat_limit_messages} messages / "
            f"{self.cfg.private_chat_limit_window_hours}h per user.\n"
            "Group chats have no per-user limit.\n"
            "Commands: /start, /status, /reset"
        )

    @staticmethod
    def _is_member_status(status: str) -> bool:
        return status in {
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.OWNER,
            ChatMemberStatus.RESTRICTED,
        }

    def _bot_was_added_to_group(self, update_obj: ChatMemberUpdated) -> bool:
        old_status = update_obj.old_chat_member.status
        new_status = update_obj.new_chat_member.status
        return (not self._is_member_status(old_status)) and self._is_member_status(
            new_status
        )

    async def _set_bot_commands(self, application: Application) -> None:
        group_commands = [
            BotCommand("start", "Show how to call bot"),
            BotCommand("status", "Show memory and model status"),
            BotCommand("reset", "Clear chat memory"),
        ]
        private_commands = [
            BotCommand("start", "Show usage"),
            BotCommand("status", "Show memory and model status"),
            BotCommand("reset", "Clear chat memory"),
        ]

        await application.bot.set_my_commands(
            private_commands,
            scope=BotCommandScopeAllPrivateChats(),
        )
        await application.bot.set_my_commands(
            group_commands,
            scope=BotCommandScopeAllGroupChats(),
        )
        await application.bot.set_my_commands(private_commands)

    async def post_init(self, application: Application) -> None:
        me = await application.bot.get_me()
        application.bot_data["bot_username"] = me.username or ""
        application.bot_data["bot_id"] = me.id
        await self._set_bot_commands(application)
        LOGGER.info("Bot identity loaded: username=@%s id=%s", me.username, me.id)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message:
            return
        bot_username = context.bot_data.get("bot_username", "")
        await message.reply_text(self._build_usage_text(bot_username))

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return

        chat_id = chat.id
        msg_count = self.store.message_count(chat_id)
        has_summary = "yes" if self.store.get_summary(chat_id) else "no"
        text = (
            f"messages={msg_count}, summary={has_summary}, "
            f"model={self.cfg.gemini_model}, summary_model={self.cfg.gemini_summary_model}, "
            f"context_tokens={self.cfg.max_context_tokens}, "
            f"recent={self.cfg.recent_messages}, group_recent={self.cfg.group_recent_messages}, "
            f"private_limit={self.cfg.private_chat_limit_messages}/{self.cfg.private_chat_limit_window_hours}h"
        )
        if not is_group_chat(chat.type) and update.effective_user:
            window_seconds = self.cfg.private_chat_limit_window_hours * 3600
            since_epoch = int(time.time()) - window_seconds
            used = self.store.count_private_requests_since(
                update.effective_user.id, since_epoch
            )
            text += f", used={used}"
        await message.reply_text(text)

    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return

        self.store.clear_chat(chat.id)
        await message.reply_text("Memory was cleared for this chat.")

    async def on_my_chat_member(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        member_update = update.my_chat_member
        if not member_update or not is_group_chat(member_update.chat.type):
            return
        if not self._bot_was_added_to_group(member_update):
            return

        bot_username = context.bot_data.get("bot_username", "")
        welcome = self._build_usage_text(bot_username)
        try:
            await context.bot.send_message(chat_id=member_update.chat.id, text=welcome)
        except Exception:
            LOGGER.exception("Failed to send group welcome message")

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not message.text:
            return

        bot_username = context.bot_data.get("bot_username", "")
        bot_id = context.bot_data.get("bot_id")
        if not should_respond(message, bot_username, self.cfg.trigger_mention, bot_id):
            return

        clean_text = sanitize_user_text(message.text, bot_username, self.cfg.trigger_mention)
        if not clean_text:
            return

        chat_id = message.chat.id
        chat_is_group = is_group_chat(message.chat.type)
        lock = self.chat_locks.setdefault(chat_id, asyncio.Lock())

        async with lock:
            if not chat_is_group:
                current_user = message.from_user
                if current_user is None:
                    return
                allowed, wait_seconds = self._check_private_rate_limit(
                    current_user.id, chat_id
                )
                if not allowed:
                    await message.reply_text(
                        "Private chat limit reached. "
                        f"Try again in {format_wait_time(wait_seconds)}."
                    )
                    return

            user_text = format_user_text(message, clean_text)
            current_user_id = message.from_user.id if message.from_user else None
            history, relevant_records = collect_context_records(
                cfg=self.cfg,
                store=self.store,
                chat_id=chat_id,
                user_text=user_text,
                is_group=chat_is_group,
                current_user_id=current_user_id,
                bot_id=bot_id if isinstance(bot_id, int) else None,
            )
            summary = self.store.get_summary(chat_id)
            model_messages = build_generation_messages(
                self.cfg,
                summary,
                history,
                user_text,
                relevant_records=relevant_records,
            )

            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                result = await self.llm.generate_from_messages(
                    model=self.cfg.gemini_model,
                    messages=model_messages,
                    system_instruction=f"{self.cfg.system_prompt}\n\n{CODE_STYLE_INSTRUCTION}",
                    max_output_tokens=self.cfg.max_reply_tokens,
                    temperature=self.cfg.temperature,
                    allow_fallbacks=True,
                )
                LOGGER.info("Generation model used: %s", result.model)
            except GeminiQuotaError:
                LOGGER.exception("Gemini quota/rate limit exceeded")
                await message.reply_text(
                    "Gemini quota/rate limit exceeded. Check project quotas and retry."
                )
                return
            except Exception:
                LOGGER.exception("Generation failed")
                await message.reply_text("Model request failed. Please try again in a moment.")
                return

            answer = (result.text or "").strip()
            if not answer:
                answer = "I could not generate a response. Please try again."

            self.store.add_message(
                chat_id,
                "user",
                user_text,
                author_id=current_user_id,
            )
            self.store.add_message(
                chat_id,
                "assistant",
                answer,
                author_id=bot_id if isinstance(bot_id, int) else None,
            )
            await self._send_reply(message, context, answer)

            try:
                await self.maybe_compact_history(chat_id)
            except Exception:
                LOGGER.exception("History compaction failed")

    async def _send_reply(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
    ) -> None:
        segments = extract_response_segments(text)
        sent_any = False

        for kind, lang, payload in segments:
            if kind == "text":
                for chunk in [part for part in split_for_telegram(payload) if part]:
                    html_chunk = markdownish_text_to_html(chunk)
                    if not sent_any:
                        await message.reply_text(
                            html_chunk,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                        sent_any = True
                    else:
                        await context.bot.send_message(
                            chat_id=message.chat.id,
                            text=html_chunk,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                continue

            for code_chunk in [part for part in split_code_for_telegram(payload) if part]:
                escaped_code = html.escape(code_chunk)
                language_attr = (
                    f' class="language-{html.escape(lang)}"' if lang else ""
                )
                html_block = f"<pre><code{language_attr}>{escaped_code}</code></pre>"
                if not sent_any:
                    await message.reply_text(html_block, parse_mode=ParseMode.HTML)
                    sent_any = True
                else:
                    await context.bot.send_message(
                        chat_id=message.chat.id,
                        text=html_block,
                        parse_mode=ParseMode.HTML,
                    )

        if not sent_any:
            await message.reply_text("No response.")

    def _check_private_rate_limit(self, user_id: int, chat_id: int) -> tuple[bool, int]:
        window_seconds = self.cfg.private_chat_limit_window_hours * 3600
        now_epoch = int(time.time())
        since_epoch = now_epoch - window_seconds
        used = self.store.count_private_requests_since(user_id, since_epoch)

        if used >= self.cfg.private_chat_limit_messages:
            oldest = self.store.oldest_private_request_since(user_id, since_epoch)
            if oldest is None:
                return False, window_seconds
            retry_after = max(1, oldest + window_seconds - now_epoch)
            return False, retry_after

        self.store.add_private_request(user_id, chat_id)
        return True, 0

    async def maybe_compact_history(self, chat_id: int) -> None:
        if self.store.message_count(chat_id) < self.cfg.summary_trigger_messages:
            return

        all_messages = self.store.fetch_all_messages(chat_id)
        if len(all_messages) <= self.cfg.recent_messages:
            return

        old_messages = all_messages[: -self.cfg.recent_messages]
        if len(old_messages) < 8:
            return

        previous_summary = self.store.get_summary(chat_id)
        dialogue_lines: list[str] = []
        consumed_tokens = 0

        for item in old_messages:
            line = f"{item['role']}: {item['content']}"
            line_tokens = approx_tokens(line)
            if consumed_tokens + line_tokens > self.cfg.summary_input_tokens:
                break
            dialogue_lines.append(line)
            consumed_tokens += line_tokens

        if not dialogue_lines:
            return

        summary_prompt = (
            "Existing summary:\n"
            f"{previous_summary or '(none)'}\n\n"
            "New dialogue fragment:\n"
            f"{chr(10).join(dialogue_lines)}\n\n"
            "Return updated summary in <= 12 short bullet points."
        )

        summary_result = await self.llm.generate_text(
            model=self.cfg.gemini_summary_model,
            prompt=summary_prompt,
            system_instruction=(
                "You maintain compact memory of a chat. Keep durable facts, user preferences, "
                "decisions, and pending tasks. Preserve who said what in group chats. "
                "Drop filler and small talk."
            ),
            max_output_tokens=self.cfg.summary_output_tokens,
            temperature=0.1,
            allow_fallbacks=True,
        )
        LOGGER.info("Summary model used: %s", summary_result.model)

        summary_text = (summary_result.text or "").strip()
        if not summary_text:
            return

        if len(summary_text) > self.cfg.summary_max_chars:
            summary_text = summary_text[: self.cfg.summary_max_chars].rstrip()

        self.store.set_summary(chat_id, summary_text)
        self.store.delete_messages_upto(chat_id, old_messages[-1]["id"])

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        LOGGER.exception("Unhandled error", exc_info=context.error)


def build_application(cfg: AppConfig) -> Application:
    bot = TelegramGeminiBot(cfg)
    app = (
        Application.builder()
        .token(cfg.telegram_bot_token)
        .post_init(bot.post_init)
        .build()
    )
    app.add_handler(ChatMemberHandler(bot.on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(CommandHandler("status", bot.cmd_status))
    app.add_handler(CommandHandler("reset", bot.cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text))
    app.add_error_handler(bot.on_error)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = AppConfig.from_env()
    app = build_application(cfg)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
