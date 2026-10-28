import unittest

import httpx

from ai_service import OpenAICompatibleService


class OpenAICompatibleTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_completions_compatible_response(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            })
        service = OpenAICompatibleService(
            api_key="test-key", base_url="https://mock.test/v1", model="mock-model",
            transcription_model="whisper-1", transport=httpx.MockTransport(handler),
        )
        try:
            result = await service.generate_from_messages(
                model="mock-model", messages=[{"role": "user", "content": "hi"}],
                system_instruction="Be helpful", max_output_tokens=50,
                temperature=0.2,
            )
            self.assertEqual(result.text, "hello")
            self.assertEqual(result.input_tokens, 10)
            self.assertEqual(result.output_tokens, 2)
        finally:
            await service.close()

    async def test_rate_limit_maps_to_quota_error(self):
        from gemini_service import GeminiQuotaError
        service = OpenAICompatibleService(
            api_key="test-key", base_url="https://mock.test/v1", model="mock-model",
            transcription_model="whisper-1",
            transport=httpx.MockTransport(lambda request: httpx.Response(429, text="slow down")),
        )
        try:
            with self.assertRaises(GeminiQuotaError):
                await service.generate_from_messages(
                    model="mock-model", messages=[{"role": "user", "content": "hi"}],
                    system_instruction=None, max_output_tokens=20, temperature=0.2,
                )
        finally:
            await service.close()


if __name__ == "__main__":
    unittest.main()
