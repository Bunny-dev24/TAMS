"""Unified LLM client: real OpenAI backend when a key is configured, falls
back to the deterministic FakeLLMBackend otherwise. All calls funnel through
here so cost logging is centralized (guardrail: cost monitoring)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from tenacity import retry, stop_after_attempt, wait_exponential

from tams.config import Settings
from tams.llm.fake import FakeLLMBackend

CostSink = Callable[[str, str, int, int, float], None]
# name/purpose, model, prompt_tokens, completion_tokens, latency_s


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient:
    def __init__(self, settings: Settings, cost_sink: CostSink | None = None):
        self.settings = settings
        self.cost_sink = cost_sink
        self.is_fake = not bool(settings.openai_api_key)
        if self.is_fake:
            self._backend = FakeLLMBackend()
            self._openai = None
        else:
            from openai import OpenAI  # imported lazily so fake mode has no hard dep

            self._openai = OpenAI(api_key=settings.openai_api_key)
            self._backend = None

    # ---------------------------------------------------------------- chat
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        purpose: str = "chat",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        fake_fn: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> ChatResult:
        t0 = time.monotonic()
        if self.is_fake:
            r = self._backend.chat(messages, model, temperature, max_tokens, fake_fn=fake_fn)
            text, pt, ct = r.text, r.prompt_tokens, r.completion_tokens
        else:
            resp = self._openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            pt = resp.usage.prompt_tokens if resp.usage else 0
            ct = resp.usage.completion_tokens if resp.usage else 0
        latency = time.monotonic() - t0
        if self.cost_sink:
            self.cost_sink(purpose, model, pt, ct, latency)
        return ChatResult(text=text, prompt_tokens=pt, completion_tokens=ct, latency_s=latency)

    # ----------------------------------------------------------- chat_json
    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: dict[str, Any],
        purpose: str = "extract",
        fake_fn: Callable[[list[dict[str, str]]], Any] | None = None,
    ) -> Any:
        t0 = time.monotonic()
        if self.is_fake:
            if fake_fn is None:
                raise ValueError("fake_fn is required in fake mode for chat_json")
            result, pt, ct = self._backend.chat_json(messages, model, schema, fake_fn)
        else:
            sys_hint = {
                "role": "system",
                "content": (
                    "Respond with STRICT JSON only, matching this schema: "
                    + json.dumps(schema)
                ),
            }
            resp = self._openai.chat.completions.create(
                model=model,
                messages=[sys_hint] + messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {}
            pt = resp.usage.prompt_tokens if resp.usage else 0
            ct = resp.usage.completion_tokens if resp.usage else 0
        latency = time.monotonic() - t0
        if self.cost_sink:
            self.cost_sink(purpose, model, pt, ct, latency)
        return result

    # ---------------------------------------------------------------- embed
    def embed(self, texts: list[str], purpose: str = "embed") -> list[list[float]]:
        if not texts:
            return []
        t0 = time.monotonic()
        if self.is_fake:
            vectors = self._backend.embed(texts, self.settings.embedding_model, self.settings.embedding_dim)
            pt = sum(len(t.split()) for t in texts)
        else:
            resp = self._openai.embeddings.create(model=self.settings.embedding_model, input=texts)
            vectors = [d.embedding for d in resp.data]
            pt = resp.usage.prompt_tokens if resp.usage else 0
        latency = time.monotonic() - t0
        if self.cost_sink:
            self.cost_sink(purpose, self.settings.embedding_model, pt, 0, latency)
        return vectors


_client: LLMClient | None = None


def get_llm_client(settings: Settings | None = None, cost_sink: CostSink | None = None) -> LLMClient:
    global _client
    if _client is None:
        from tams.config import settings as default_settings

        _client = LLMClient(settings or default_settings, cost_sink=cost_sink)
    return _client


def reset_llm_client() -> None:
    """Used by tests to force a fresh client (e.g. after monkeypatching env)."""
    global _client
    _client = None
