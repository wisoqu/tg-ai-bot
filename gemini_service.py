import asyncio
import logging
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class GeminiQuotaError(RuntimeError):
    pass


class GeminiService:
    AUTO_MODEL = "auto"
    # Auto mode intentionally stays on economical Flash-Lite models. It never
    # silently falls back to a potentially billable Pro model.
    DEFAULT_AUTO_MODELS = [
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    def __init__(self, api_key: str, fallback_models: list[str] | None = None, api_keys: list[str] | None = None) -> None:
        self.api_keys = list(dict.fromkeys(api_keys or [api_key]))
        self._key_index = 0
        self.client = genai.Client(api_key=self.api_keys[0])
        self.fallback_models = [
            self._normalize_model_name(item)
            for item in (fallback_models or [])
            if self._normalize_model_name(item)
        ]
        self._available_models: set[str] | None = None
        self._list_lock = asyncio.Lock()
        self._cooldown_until: dict[tuple[int, str], float] = {}

    async def generate_from_messages(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        system_instruction: str | None,
        max_output_tokens: int,
        temperature: float,
        allow_fallbacks: bool = True,
    ) -> GeminiResult:
        prompt = self._messages_to_prompt(messages)
        return await self.generate_text(
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            allow_fallbacks=allow_fallbacks,
        )

    async def transcribe_audio(self, audio: bytes, mime_type: str, model: str = "gemini-2.5-flash-lite", duration_seconds: int = 0) -> GeminiResult:
        part = types.Part.from_bytes(data=audio, mime_type=mime_type)
        last_error: Exception | None = None
        response = None
        for attempt in range(max(1, len(self.api_keys))):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model, contents=[part, "Transcribe the spoken words verbatim. Return only the transcription; preserve the original language."]
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < len(self.api_keys):
                    self._rotate_key()
                else:
                    raise
        if response is None:
            raise RuntimeError("Gemini transcription failed") from last_error
        text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        return GeminiResult(text=text, model=model,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0))

    async def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str | None,
        max_output_tokens: int,
        temperature: float,
        allow_fallbacks: bool = True,
    ) -> GeminiResult:
        all_candidates = await self._build_candidate_models(model, allow_fallbacks)
        candidate_models = all_candidates
        if not candidate_models:
            raise GeminiQuotaError("No Gemini models are currently available")

        last_error: Exception | None = None
        attempted = False
        for candidate_model in candidate_models:
            # A quota is per Google project, not per key, but a configured key may
            # belong to another project. Retry the same economical model with each key.
            key_attempts = len(self.api_keys)
            for _ in range(key_attempts):
                key_index = self._key_index
                if self._cooldown_until.get((key_index, candidate_model), 0) > time.time():
                    self._rotate_key()
                    continue
                client = self.client  # stable for this request even if another task rotates keys
                attempted = True
                try:
                    response = await self._generate_with_model(
                        candidate_model=candidate_model, prompt=prompt,
                        system_instruction=system_instruction,
                        max_output_tokens=max_output_tokens, temperature=temperature,
                        client=client,
                    )
                    text = (response.text or "").strip()
                    if text:
                        usage = getattr(response, "usage_metadata", None)
                        return GeminiResult(
                            text=text, model=candidate_model,
                            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0)
                            + int(getattr(usage, "thoughts_token_count", 0) or 0),
                        )
                    last_error = RuntimeError(f"{candidate_model} returned empty output")
                    break
                except errors.ClientError as exc:
                    last_error = exc
                    if self._is_quota_error(exc):
                        retry_seconds = self._parse_retry_seconds(exc)
                        self._cooldown_until[(key_index, candidate_model)] = time.time() + retry_seconds
                        LOGGER.warning("Gemini model %s quota-limited on key slot %s; cooldown %ss",
                                       candidate_model, key_index + 1, retry_seconds)
                        self._rotate_key()
                        continue
                    if self._is_key_error(exc):
                        self._rotate_key()
                        continue
                    raise
                except Exception as exc:
                    last_error = exc
                    if self._is_key_error(exc):
                        self._rotate_key()
                        continue
                    break

        if last_error and self._is_quota_error(last_error):
            raise GeminiQuotaError("Gemini quota/rate limit exceeded") from last_error
        if last_error:
            raise RuntimeError("Gemini generation failed") from last_error
        if not attempted:
            raise GeminiQuotaError("All configured Gemini key/model pairs are in cooldown")
        raise RuntimeError("Gemini generation returned empty output")

    async def _generate_with_model(
        self,
        *,
        candidate_model: str,
        prompt: str,
        system_instruction: str | None,
        max_output_tokens: int,
        temperature: float,
        client=None,
    ):
        client = client or self.client
        config = self._build_generation_config(
            model_name=candidate_model,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            disable_thinking=True,
        )
        try:
            return await client.aio.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=config,
            )
        except errors.ClientError as exc:
            if self._is_thinking_not_supported_error(exc):
                fallback_config = self._build_generation_config(
                    model_name=candidate_model,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    disable_thinking=False,
                )
                return await client.aio.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                    config=fallback_config,
                )
            raise

    def _build_generation_config(
        self,
        *,
        model_name: str,
        system_instruction: str | None,
        max_output_tokens: int,
        temperature: float,
        disable_thinking: bool,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, object] = {
            "system_instruction": system_instruction,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if disable_thinking and self._should_disable_thinking(model_name):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    async def _build_candidate_models(
        self, model: str, allow_fallbacks: bool
    ) -> list[str]:
        base = self._normalize_model_name(model)
        if not base or base.lower() == self.AUTO_MODEL:
            models = await self._resolve_auto_models()
        else:
            models = [base]

        if allow_fallbacks:
            models.extend(self.fallback_models)
            models.extend(await self._resolve_auto_models())

        deduped: list[str] = []
        seen: set[str] = set()
        for item in models:
            if item and item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    async def _resolve_auto_models(self) -> list[str]:
        available = await self._load_available_models()
        if not available:
            return list(self.DEFAULT_AUTO_MODELS)

        resolved: list[str] = []
        for preferred in self.DEFAULT_AUTO_MODELS:
            if preferred in available:
                resolved.append(preferred)

        # Do not append every model visible to the key: that could silently pick
        # a paid/pro model after an inexpensive model is rate-limited.
        return resolved or list(self.DEFAULT_AUTO_MODELS)

    async def _load_available_models(self) -> set[str]:
        if self._available_models is not None:
            return self._available_models

        async with self._list_lock:
            if self._available_models is not None:
                return self._available_models

            found: set[str] = set()
            try:
                pager = await self.client.aio.models.list()
                async for item in pager:
                    name = self._normalize_model_name(getattr(item, "name", ""))
                    actions = getattr(item, "supported_actions", None) or []
                    if name and "generateContent" in actions:
                        found.add(name)
            except Exception:
                LOGGER.exception("Failed to list Gemini models")

            self._available_models = found
            return found

    def _filter_cooldown_models(self, models: list[str]) -> list[str]:
        if not models:
            return []

        now = time.time()
        ready = [m for m in models if self._cooldown_until.get(m, 0) <= now]
        if ready:
            return ready

        soonest = min(models, key=lambda m: self._cooldown_until.get(m, 0))
        return [soonest]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for item in messages:
            role = item.get("role", "user").upper()
            content = item.get("content", "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _rotate_key(self) -> bool:
        if len(self.api_keys) < 2:
            return False
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self._key_index])
        LOGGER.warning("Switched Gemini API key index %s; keys from one project share quota", self._key_index + 1)
        return True

    @staticmethod
    def _is_key_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(term in text for term in ("api key not valid", "invalid api key", "permission_denied", "unauthenticated"))

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        cleaned = (name or "").strip()
        if cleaned.startswith("models/"):
            return cleaned.split("/", 1)[1]
        return cleaned

    @staticmethod
    def _parse_retry_seconds(exc: Exception) -> int:
        text = str(exc)

        match_retry = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
        if match_retry:
            return max(10, int(float(match_retry.group(1))) + 1)

        match_delay = re.search(r"retryDelay['\"]?:\s*['\"]?([0-9]+)s", text, flags=re.IGNORECASE)
        if match_delay:
            return max(10, int(match_delay.group(1)) + 1)

        return 30

    @staticmethod
    def _should_disable_thinking(model_name: str) -> bool:
        return model_name.startswith("gemini-")

    @staticmethod
    def _is_thinking_not_supported_error(exc: errors.ClientError) -> bool:
        text = str(exc).upper()
        return ("THINKING IS NOT ENABLED" in text) or ("THINKING" in text and "INVALID_ARGUMENT" in text)

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        if isinstance(exc, errors.ClientError):
            code = getattr(exc, "code", None)
            status = str(getattr(exc, "status", "")).upper()
            if code == 429:
                return True
            if "RESOURCE_EXHAUSTED" in status:
                return True

        text = str(exc).upper()
        return ("429" in text) or ("RESOURCE_EXHAUSTED" in text) or ("QUOTA" in text)
