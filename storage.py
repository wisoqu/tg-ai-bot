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
        if path != ":memory:":
            from pathlib import Path
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
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
            conn.execute("""CREATE TABLE IF NOT EXISTS usage_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL,
                scope_id INTEGER NOT NULL, tokens INTEGER NOT NULL,
                created_at INTEGER NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS usage_reservations (
                reservation_id TEXT PRIMARY KEY, scope TEXT NOT NULL, scope_id INTEGER NOT NULL,
                reserved_tokens INTEGER NOT NULL, paid_reserved INTEGER NOT NULL,
                user_id INTEGER, created_at INTEGER NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_scope_time ON usage_ledger(scope, scope_id, created_at)")
            conn.execute("""CREATE TABLE IF NOT EXISTS saved_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL, fact TEXT NOT NULL,
                created_at TEXT NOT NULL, expires_at INTEGER NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_saved_memories_chat_exp ON saved_memories(chat_id, expires_at)")
            conn.execute("""CREATE TABLE IF NOT EXISTS known_chats (
                chat_id INTEGER PRIMARY KEY, chat_type TEXT NOT NULL, title TEXT NOT NULL DEFAULT '')""")
            conn.execute("""CREATE TABLE IF NOT EXISTS bot_settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS token_balances (
                user_id INTEGER PRIMARY KEY, tokens INTEGER NOT NULL DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS star_purchases (
                payload TEXT PRIMARY KEY, user_id INTEGER NOT NULL, tokens INTEGER NOT NULL,
                stars INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                telegram_charge_id TEXT UNIQUE, created_at INTEGER NOT NULL)""")
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

    def create_purchase(self, payload: str, user_id: int, tokens: int, stars: int) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO star_purchases(payload,user_id,tokens,stars,created_at) VALUES(?,?,?,?,?)",(payload,user_id,tokens,stars,int(time.time())))

    def purchase(self, payload: str) -> dict[str, Any] | None:
        with self._connect() as conn: row=conn.execute("SELECT * FROM star_purchases WHERE payload=?",(payload,)).fetchone()
        return dict(row) if row else None

    def settle_purchase(self, payload: str, user_id: int, stars: int, charge_id: str) -> bool:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM star_purchases WHERE payload=?", (payload,)).fetchone()
            if not row or row["status"] != "pending" or row["user_id"] != user_id or row["stars"] != stars:
                conn.rollback()
                return False
            conn.execute("UPDATE star_purchases SET status='paid',telegram_charge_id=? WHERE payload=?", (charge_id, payload))
            conn.execute("INSERT INTO token_balances(user_id,tokens) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET tokens=tokens+excluded.tokens", (user_id, row["tokens"]))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            return False
        finally:
            conn.close()

    def token_balance(self, user_id: int) -> int:
        with self._connect() as conn: row=conn.execute("SELECT tokens FROM token_balances WHERE user_id=?",(user_id,)).fetchone()
        return int(row["tokens"]) if row else 0

    def spend_balance(self, user_id: int, tokens: int) -> bool:
        with self._connect() as conn:
            row=conn.execute("SELECT tokens FROM token_balances WHERE user_id=?",(user_id,)).fetchone()
            balance=int(row["tokens"]) if row else 0
            if balance < tokens: return False
            conn.execute("UPDATE token_balances SET tokens=tokens-? WHERE user_id=?",(tokens,user_id))
            return True

    def add_usage(self, scope: str, scope_id: int, tokens: int, created_at: int | None = None) -> None:
        if tokens <= 0: return
        with self._connect() as conn:
            timestamp = created_at or int(time.time())
            conn.execute("INSERT INTO usage_ledger(scope,scope_id,tokens,created_at) VALUES(?,?,?,?)",
                         (scope, scope_id, tokens, timestamp))
            if scope != "global":
                conn.execute("INSERT INTO usage_ledger(scope,scope_id,tokens,created_at) VALUES('global',0,?,?)",
                             (tokens, timestamp))

    def reserve_usage(
        self, reservation_id: str, scope: str, scope_id: int, tokens: int,
        daily_limit: int, global_limit: int, *, since: int, user_id: int | None = None,
    ) -> tuple[bool, str]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM usage_reservations WHERE created_at<?", (int(time.time())-900,))
            used_row = conn.execute("SELECT COALESCE(SUM(tokens),0) n FROM usage_ledger WHERE scope=? AND scope_id=? AND created_at>=?",(scope,scope_id,since)).fetchone()
            reserved_scope = conn.execute("SELECT COALESCE(SUM(reserved_tokens-paid_reserved),0) n FROM usage_reservations WHERE scope=? AND scope_id=? AND created_at>=?",(scope,scope_id,since)).fetchone()
            free_left = max(0, daily_limit-int(used_row["n"])-int(reserved_scope["n"]))
            global_used = conn.execute("SELECT COALESCE(SUM(tokens),0) n FROM usage_ledger WHERE scope='global' AND created_at>=?",(since,)).fetchone()
            global_reserved = conn.execute("SELECT COALESCE(SUM(reserved_tokens),0) n FROM usage_reservations WHERE created_at>=?",(since,)).fetchone()
            if int(global_used["n"])+int(global_reserved["n"])+tokens > global_limit:
                conn.rollback()
                return False, "global"
            paid_due=max(0,tokens-free_left)
            if paid_due:
                if user_id is None:
                    conn.rollback()
                    return False, "quota"
                balance_row=conn.execute("SELECT tokens FROM token_balances WHERE user_id=?",(user_id,)).fetchone()
                balance=int(balance_row["tokens"]) if balance_row else 0
                pending=conn.execute("SELECT COALESCE(SUM(paid_reserved),0) n FROM usage_reservations WHERE user_id=? AND created_at>=?",(user_id,since)).fetchone()
                if paid_due > balance-int(pending["n"]):
                    conn.rollback()
                    return False, "quota"
            conn.execute("INSERT INTO usage_reservations(reservation_id,scope,scope_id,reserved_tokens,paid_reserved,user_id,created_at) VALUES(?,?,?,?,?,?,?)",(reservation_id,scope,scope_id,tokens,paid_due,user_id,int(time.time())))
            conn.commit()
            return True, ""
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def settle_reservation(self, reservation_id: str, tokens: int) -> tuple[bool, int]:
        conn=self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            reservation=conn.execute("SELECT * FROM usage_reservations WHERE reservation_id=?",(reservation_id,)).fetchone()
            if not reservation:
                conn.rollback()
                return False, 0
            paid_free_capacity=max(0,int(reservation["reserved_tokens"])-int(reservation["paid_reserved"]))
            paid_due=max(0,tokens-paid_free_capacity)
            user_id=reservation["user_id"]
            if paid_due:
                balance_row=conn.execute("SELECT tokens FROM token_balances WHERE user_id=?",(user_id,)).fetchone()
                balance=int(balance_row["tokens"]) if balance_row else 0
                if paid_due>balance:
                    conn.rollback()
                    return False, paid_due
                conn.execute("UPDATE token_balances SET tokens=tokens-? WHERE user_id=?",(paid_due,user_id))
            timestamp=int(time.time())
            conn.execute("INSERT INTO usage_ledger(scope,scope_id,tokens,created_at) VALUES(?,?,?,?)",(reservation["scope"],reservation["scope_id"],tokens,timestamp))
            conn.execute("INSERT INTO usage_ledger(scope,scope_id,tokens,created_at) VALUES('global',0,?,?)",(tokens,timestamp))
            conn.execute("DELETE FROM usage_reservations WHERE reservation_id=?",(reservation_id,))
            conn.commit()
            return True,paid_due
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def release_reservation(self, reservation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM usage_reservations WHERE reservation_id=?",(reservation_id,))

    def consume_usage(
        self, scope: str, scope_id: int, tokens: int, daily_limit: int,
        *, since: int, user_id: int | None = None, global_limit: int | None = None,
    ) -> tuple[bool, int]:
        """Atomically record usage and debit paid balance only above the free quota."""
        if tokens <= 0:
            return True, 0
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens),0) AS used FROM usage_ledger "
                "WHERE scope=? AND scope_id=? AND created_at>=?",
                (scope, scope_id, since),
            ).fetchone()
            used = int(row["used"])
            if global_limit is not None:
                global_row = conn.execute("SELECT COALESCE(SUM(tokens),0) AS used FROM usage_ledger WHERE scope='global' AND created_at>=?", (since,)).fetchone()
                if int(global_row["used"]) + tokens > global_limit:
                    conn.rollback()
                    return False, 0
            paid_due = max(0, tokens - max(0, daily_limit - used))
            balance = 0
            if user_id is not None:
                balance_row = conn.execute(
                    "SELECT tokens FROM token_balances WHERE user_id=?", (user_id,)
                ).fetchone()
                balance = int(balance_row["tokens"]) if balance_row else 0
            if paid_due > balance:
                conn.rollback()
                return False, paid_due
            if paid_due:
                conn.execute(
                    "UPDATE token_balances SET tokens=tokens-? WHERE user_id=?",
                    (paid_due, user_id),
                )
            timestamp = int(time.time())
            conn.execute(
                "INSERT INTO usage_ledger(scope,scope_id,tokens,created_at) VALUES(?,?,?,?)",
                (scope, scope_id, tokens, timestamp),
            )
            if scope != "global":
                conn.execute("INSERT INTO usage_ledger(scope,scope_id,tokens,created_at) VALUES('global',0,?,?)", (tokens, timestamp))
            conn.commit()
            return True, paid_due
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def usage_since(self, scope: str, scope_id: int, since: int) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(SUM(tokens),0) n FROM usage_ledger WHERE scope=? AND scope_id=? AND created_at>=?", (scope,scope_id,since)).fetchone()
        return int(row["n"])

    def remember(self, chat_id: int, user_id: int, fact: str, retention_days: int = 30) -> None:
        if not fact.strip() or len(fact) > 500:
            raise ValueError("Saved facts must contain 1-500 characters")
        now = int(time.time())
        with self._connect() as conn:
            conn.execute("INSERT INTO saved_memories(chat_id,user_id,fact,created_at,expires_at) VALUES(?,?,?,?,?)",
                         (chat_id,user_id,fact,datetime.now(timezone.utc).isoformat(),now+retention_days*86400))
            conn.execute("DELETE FROM saved_memories WHERE chat_id=? AND id NOT IN (SELECT id FROM saved_memories WHERE chat_id=? ORDER BY id DESC LIMIT 15)", (chat_id,chat_id))

    def fetch_memories(self, chat_id: int) -> list[dict[str, Any]]:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute("DELETE FROM saved_memories WHERE expires_at<=?", (now,))
            rows = conn.execute("SELECT id,user_id,fact,created_at FROM saved_memories WHERE chat_id=? AND expires_at>? ORDER BY id DESC LIMIT 15",(chat_id,now)).fetchall()
        return [dict(row) for row in rows]

    def clear_memories(self, chat_id: int) -> None:
        with self._connect() as conn: conn.execute("DELETE FROM saved_memories WHERE chat_id=?",(chat_id,))

    def remember_chat(self, chat_id: int, chat_type: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO known_chats(chat_id,chat_type,title) VALUES(?,?,?) ON CONFLICT(chat_id) DO UPDATE SET chat_type=excluded.chat_type,title=excluded.title",(chat_id,chat_type,title))

    def known_chat_ids(self) -> list[int]:
        with self._connect() as conn: rows=conn.execute("SELECT chat_id FROM known_chats").fetchall()
        return [int(r["chat_id"]) for r in rows]

    def get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as conn: row=conn.execute("SELECT value FROM bot_settings WHERE key=?",(key,)).fetchone()
        return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn: conn.execute("INSERT INTO bot_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(key,value))

    def prune_expired(self, retention_days: int = 30) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - retention_days*86400
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE created_at < ?", (datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),))
            # Legacy rolling summaries had no reliable source-date range; remove them to honor TTL.
            conn.execute("DELETE FROM chat_state")
            conn.execute("DELETE FROM usage_ledger WHERE created_at < ?", (int(time.time())-90*86400,))
            conn.execute("DELETE FROM saved_memories WHERE expires_at<=?",(int(time.time()),))
            conn.execute("DELETE FROM star_purchases WHERE status='pending' AND created_at<?", (int(time.time())-30*86400,))
            conn.execute("DELETE FROM usage_reservations WHERE created_at<?", (int(time.time())-900,))

    def clear_all_memory(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM chat_state")
            conn.execute("DELETE FROM saved_memories")
            conn.commit()

    def vacuum(self) -> None:
        conn = self._connect()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def clear_chat(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chat_state WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM saved_memories WHERE chat_id = ?", (chat_id,))
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
