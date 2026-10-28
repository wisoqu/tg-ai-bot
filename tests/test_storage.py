import tempfile
import unittest
from pathlib import Path

from storage import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "nested" / "bot.sqlite3")
        self.store = MemoryStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_parent_directory_and_expires_data(self):
        self.store.add_message(101, "user", "old message")
        self.store.remember(101, 7, "short-lived fact", retention_days=1)
        with self.store._connect() as conn:
            conn.execute("UPDATE saved_memories SET expires_at=0 WHERE chat_id=101")
        self.store.prune_expired(retention_days=0)
        self.assertEqual(self.store.message_count(101), 0)
        self.assertEqual(self.store.fetch_memories(101), [])

    def test_usage_reservations_respect_user_and_global_limits(self):
        # Credit a simulated Stars purchase.
        self.store.create_purchase("invoice-one", 7, 1_000, 10)
        self.assertTrue(self.store.settle_purchase("invoice-one", 7, 10, "charge-one"))
        self.assertFalse(self.store.settle_purchase("invoice-one", 7, 10, "charge-one"))
        self.assertEqual(self.store.token_balance(7), 1_000)

        ok, reason = self.store.reserve_usage(
            "r1", "user", 7, 80, 100, 1_000, since=0, user_id=7
        )
        self.assertTrue(ok, reason)
        ok, reason = self.store.reserve_usage(
            "r2", "group", -100, 950, 1_000, 1_000, since=0
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "global")

        settled, paid = self.store.settle_reservation("r1", 70)
        self.assertTrue(settled)
        self.assertEqual(paid, 0)
        self.assertEqual(self.store.usage_since("user", 7, 0), 70)
        self.assertEqual(self.store.usage_since("global", 0, 0), 70)

    def test_paid_balance_is_reserved_and_debited_atomically(self):
        self.store.create_purchase("invoice-two", 7, 500, 5)
        self.store.settle_purchase("invoice-two", 7, 5, "charge-two")
        # 80 free tokens remain in the daily quota; the other 20 are reserved.
        self.store.add_usage("user", 7, 20, created_at=10)
        ok, reason = self.store.reserve_usage(
            "paid-request", "user", 7, 100, 100, 1_000, since=0, user_id=7
        )
        self.assertTrue(ok, reason)
        settled, paid = self.store.settle_reservation("paid-request", 90)
        self.assertTrue(settled)
        self.assertEqual(paid, 10)
        self.assertEqual(self.store.token_balance(7), 490)

    def test_manual_reset_preserves_payments(self):
        self.store.add_message(9, "user", "private")
        self.store.remember(9, 3, "remember me")
        self.store.clear_all_memory()
        self.assertEqual(self.store.message_count(9), 0)
        self.assertEqual(self.store.fetch_memories(9), [])


if __name__ == "__main__":
    unittest.main()
