import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any


_WORD_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]{3,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "you",
    "your",
    "как",
    "что",
    "это",
    "или",
    "для",
    "чтобы",
    "если",
    "тебя",
    "твой",
    "мне",
    "про",
}


def _tokenize_query(text: str) -> list[str]:
    tokens = [item.lower() for item in _WORD_RE.findall(text or "")]
    filtered = [item for item in tokens if item not in _STOPWORDS]
    # preserve order while deduplicating
    seen: set[str] = set()
    unique: list[str] = []
    for token in filtered:
        if token not in seen:
            unique.append(token)
            seen.add(token)
    return unique


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table_name: str, column_name: str, definition: str
    ) -> None:
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        names = {row["name"] for row in columns}
        if column_name not in names:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "messages", "author_id", "INTEGER")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id
                ON messages(chat_id, id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_chat_author_id_id
                ON messages(chat_id, author_id, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS private_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_private_requests_user_created
                ON private_requests(user_id, created_at)
                """
            )
            conn.commit()

    def add_message(
        self, chat_id: int, role: str, content: str, author_id: int | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(chat_id, role, content, author_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    role,
                    content,
                    author_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def fetch_recent_messages(self, chat_id: int, limit: int) -> list[dict[str, str]]:
        records = self.fetch_recent_records(chat_id, limit)
        return [{"role": row["role"], "content": row["content"]} for row in records]

    def fetch_recent_records(self, chat_id: int, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, author_id
                FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()

        rows = list(reversed(rows))
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "author_id": row["author_id"],
            }
            for row in rows
        ]

    def fetch_recent_by_authors(
        self, chat_id: int, author_ids: list[int], limit: int
    ) -> list[dict[str, Any]]:
        if not author_ids:
            return []

        placeholders = ", ".join("?" for _ in author_ids)
        params: list[Any] = [chat_id, *author_ids, limit]
        query = (
            "SELECT id, role, content, author_id "
            "FROM messages "
            "WHERE chat_id = ? AND author_id IN (" + placeholders + ") "
            "ORDER BY id DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        rows = list(reversed(rows))
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "author_id": row["author_id"],
            }
            for row in rows
        ]

    def fetch_relevant_records(
        self,
        *,
        chat_id: int,
        query: str,
        limit: int,
        lookback_messages: int,
        prefer_author_id: int | None = None,
        bot_author_id: int | None = None,
        exclude_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        tokens = _tokenize_query(query)
        if not tokens or limit <= 0 or lookback_messages <= 0:
            return []

        with self._connect() as conn:
            candidates = conn.execute(
                """
                SELECT id, role, content, author_id
                FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, lookback_messages),
            ).fetchall()

        blocked_ids = exclude_ids or set()
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for row in candidates:
            msg_id = int(row["id"])
            if msg_id in blocked_ids:
                continue
            content = (row["content"] or "").lower()
            overlap = sum(1 for token in tokens if token in content)
            if overlap == 0:
                continue

            score = overlap * 3
            author_id = row["author_id"]
            if prefer_author_id is not None and author_id == prefer_author_id:
                score += 2
            if bot_author_id is not None and author_id == bot_author_id:
                score += 1
            if row["role"] == "assistant":
                score += 1

            scored.append(
                (
                    score,
                    msg_id,
                    {
                        "id": msg_id,
                        "role": row["role"],
                        "content": row["content"],
                        "author_id": author_id,
                    },
                )
            )

        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], -item[1]))
        top_records = [item[2] for item in scored[:limit]]
        top_records.sort(key=lambda item: int(item["id"]))
        return top_records

    def fetch_all_messages(self, chat_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, author_id
                FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (chat_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "author_id": row["author_id"],
            }
            for row in rows
        ]

    def message_count(self, chat_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_summary(self, chat_id: int) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM chat_state WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def set_summary(self, chat_id: int, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_state(chat_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE
                SET summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (chat_id, summary, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def delete_messages_upto(self, chat_id: int, max_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE chat_id = ? AND id <= ?",
                (chat_id, max_id),
            )
            conn.commit()

    def clear_chat(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chat_state WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM private_requests WHERE chat_id = ?", (chat_id,))
            conn.commit()

    def add_private_request(self, user_id: int, chat_id: int) -> None:
        now_epoch = int(time.time())
        month_ago = now_epoch - (30 * 24 * 3600)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO private_requests(user_id, chat_id, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, chat_id, now_epoch),
            )
            conn.execute(
                "DELETE FROM private_requests WHERE created_at < ?",
                (month_ago,),
            )
            conn.commit()

    def count_private_requests_since(self, user_id: int, since_epoch: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM private_requests
                WHERE user_id = ? AND created_at >= ?
                """,
                (user_id, since_epoch),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def oldest_private_request_since(
        self, user_id: int, since_epoch: int
    ) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(created_at) AS oldest
                FROM private_requests
                WHERE user_id = ? AND created_at >= ?
                """,
                (user_id, since_epoch),
            ).fetchone()
        if not row or row["oldest"] is None:
            return None
        return int(row["oldest"])
