"""Provider abstraction for Gemini and OpenAI-compatible chat APIs."""
from __future__ import annotations

import logging
import mimetypes
from typing import Any

import httpx

from gemini_service import GeminiQuotaError, GeminiResult, GeminiService

LOGGER = logging.getLogger(__name__)


class OpenAICompatibleService:
    """Works with OpenAI Chat Completions-compatible providers and gateways."""

    def __init__(self, *, api_key: str, base_url: str, model: str, transcription_model: str, transport=None):
        self.model = model
        self.transcription_model = transcription_model
        normalized = base_url.rstrip("/") + "/"
        self.client = httpx.AsyncClient(
            base_url=normalized,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(90.0, connect=15.0),
            transport=transport,
        )

    async def generate_from_messages(
        self, *, model: str, messages: list[dict[str, str]],
        system_instruction: str | None, max_output_tokens: int,
        temperature: float, allow_fallbacks: bool = True,
    ) -> GeminiResult:
        chat_messages: list[dict[str, str]] = []
        if system_instruction:
            chat_messages.append({"role": "system", "content": system_instruction})
        for item in messages:
            role = item.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = item.get("content", "").strip()
            if content:
                chat_messages.append({"role": role, "content": content})
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": chat_messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        response = await self._post_json("chat/completions", payload)
        try:
            choice = response["choices"][0]["message"]["content"]
            text = "".join(part.get("text", "") for part in choice) if isinstance(choice, list) else str(choice or "")
            usage = response.get("usage") or {}
            return GeminiResult(
                text=text.strip(), model=payload["model"],
                input_tokens=int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                output_tokens=int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The compatible API returned an unexpected chat-completions response") from exc

    async def generate_text(
        self, *, model: str, prompt: str, system_instruction: str | None,
        max_output_tokens: int, temperature: float, allow_fallbacks: bool = True,
    ) -> GeminiResult:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_from_messages(
            model=model, messages=messages, system_instruction=system_instruction,
            max_output_tokens=max_output_tokens, temperature=temperature,
            allow_fallbacks=allow_fallbacks,
        )

    async def transcribe_audio(
        self, audio: bytes, mime_type: str, model: str | None = None,
        duration_seconds: int = 0,
    ) -> GeminiResult:
        filename = "voice" + (mimetypes.guess_extension(mime_type) or ".ogg")
        try:
            response = await self.client.post(
                "audio/transcriptions",
                data={"model": model or self.transcription_model},
                files={"file": (filename, audio, mime_type)},
            )
            self._raise_for_status(response)
            data = response.json()
            text = str(data.get("text", "")).strip()
            # Transcription endpoints usually charge by audio duration rather than
            # text tokens. Count an approximate audio-token equivalent in app quotas.
            estimated_audio_tokens = max(0, int(duration_seconds) * 32)
            return GeminiResult(text=text, model=model or self.transcription_model,
                                input_tokens=estimated_audio_tokens, output_tokens=0)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Transcription endpoint failed: {exc}") from exc

    async def check_connection(self) -> str:
        response = await self.client.get("models")
        self._raise_for_status(response)
        body = response.json()
        models = body.get("data", []) if isinstance(body, dict) else []
        return f"OpenAI-compatible endpoint OK ({len(models)} models listed; configured: {self.model})"

    async def close(self) -> None:
        await self.client.aclose()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(path, json=payload)
            if response.status_code == 400 and "max_tokens" in response.text and "max_completion_tokens" in response.text:
                payload = dict(payload)
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                response = await self.client.post(path, json=payload)
            self._raise_for_status(response)
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("The compatible API response must be a JSON object")
            return body
        except httpx.HTTPError as exc:
            raise RuntimeError(f"AI provider request failed: {exc}") from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 429:
            raise GeminiQuotaError("AI provider rate/quota limit reached (HTTP 429)")
        if response.is_error:
            detail = response.text[:500]
            raise RuntimeError(f"AI provider returned HTTP {response.status_code}: {detail}")


def create_ai_service(cfg):
    if cfg.ai_provider == "gemini":
        return GeminiService(
            api_key=cfg.ai_api_key,
            api_keys=cfg.gemini_api_keys,
            fallback_models=cfg.gemini_fallback_models,
        )
    return OpenAICompatibleService(
        api_key=cfg.ai_api_key,
        base_url=cfg.openai_compat_base_url,
        model=cfg.ai_model,
        transcription_model=cfg.transcription_model,
    )
