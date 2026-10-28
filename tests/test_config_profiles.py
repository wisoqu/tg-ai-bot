import os
import unittest
from unittest.mock import patch

from config import AppConfig


class ConfigProfilesTests(unittest.TestCase):
    def test_gemini_accepts_space_separated_keys(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "123:telegram",
            "GEMINI_API_KEYS": "key-one key-two",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AppConfig.from_env()
        self.assertEqual(cfg.ai_provider, "gemini")
        self.assertEqual(cfg.ai_api_key, "key-one")
        self.assertEqual(cfg.gemini_api_keys, ["key-one", "key-two"])

    def test_openai_compatible_provider_does_not_require_gemini_key(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "123:telegram",
            "AI_PROVIDER": "openai-compatible",
            "AI_API_KEY": "provider-key",
            "OPENAI_COMPAT_BASE_URL": "https://api.example.test/v1",
            "OPENAI_COMPAT_MODEL": "chat-model",
            "TRANSCRIPTION_MODEL": "",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AppConfig.from_env()
        self.assertEqual(cfg.ai_api_key, "provider-key")
        self.assertEqual(cfg.ai_model, "chat-model")
        self.assertEqual(cfg.transcription_model, "")


if __name__ == "__main__":
    unittest.main()
