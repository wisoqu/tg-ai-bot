import unittest
from datetime import datetime, timezone

from main import (
    build_generation_messages,
    contains_trigger_mention,
    sanitize_user_text,
    should_respond,
)
from telegram import Message


class MainHelperTests(unittest.TestCase):
    def _message(self, text, *, group=True, reply_user=None, entities=None):
        data = {
            "message_id": 7,
            "date": int(datetime.now(timezone.utc).timestamp()),
            "chat": {"id": -100123, "type": "group" if group else "private"},
            "from": {"id": 22, "is_bot": False, "first_name": "Tester"},
            "text": text,
        }
        if entities:
            data["entities"] = entities
        if reply_user is not None:
            data["reply_to_message"] = {
                "message_id": 6,
                "date": data["date"],
                "chat": data["chat"],
                "from": {"id": reply_user, "is_bot": reply_user == 99, "first_name": "Bot"},
                "text": "previous response",
            }
        return Message.de_json(data, bot=None)

    def test_group_mentions_and_replies_trigger(self):
        mention = self._message("@helper hello", entities=[{"type": "mention", "offset": 0, "length": 7}])
        reply = self._message("continue", reply_user=99)
        ambient = self._message("ordinary group conversation")
        self.assertTrue(should_respond(mention, "helper", "@ai", 99))
        self.assertTrue(should_respond(reply, "helper", "@ai", 99))
        self.assertFalse(should_respond(ambient, "helper", "@ai", 99))

    def test_trigger_and_sanitization(self):
        self.assertTrue(contains_trigger_mention("please @ai help", "@ai"))
        self.assertEqual(sanitize_user_text("@helper: explain @ai this", "helper", "@ai"), "explain this")

    def test_context_preserves_recent_turn_order(self):
        class Config:
            system_prompt = "helpful"
            max_context_tokens = 1000
            retrieval_max_tokens = 100
        messages = build_generation_messages(
            Config(), "", [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ], "third",
        )
        self.assertEqual([m["content"] for m in messages], ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
