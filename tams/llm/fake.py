"""Deterministic, zero-network LLM backend used when no OPENAI_API_KEY is set.

Lets the whole pipeline run (dev, CI, tests) without a key or network access.
Embeddings are hash-seeded deterministic unit vectors so cosine similarity is
stable and meaningful across runs (same text -> same vector).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np


class FakeChatResult:
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeLLMBackend:
    """Stub backend: no network, no API key required."""

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        fake_fn: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> FakeChatResult:
        prompt = "\n".join(m["content"] for m in messages)
        if fake_fn is not None:
            text = fake_fn(messages)
        else:
            # Generic fallback: echo a short deterministic acknowledgement.
            text = "[fake-llm] " + prompt[-200:]
        return FakeChatResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: dict[str, Any],
        fake_fn: Callable[[list[dict[str, str]]], dict | list],
    ) -> tuple[dict | list, int, int]:
        """fake_fn is REQUIRED here — callers supply deterministic extraction
        logic (regex/heuristics) so structured extraction still behaves
        sensibly without a real model."""
        result = fake_fn(messages)
        prompt = "\n".join(m["content"] for m in messages)
        completion = json.dumps(result)
        return result, len(prompt.split()), len(completion.split())

    def embed(self, texts: list[str], model: str, dim: int = 1536) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=dim)
            v = v / (np.linalg.norm(v) + 1e-9)
            vectors.append(v.tolist())
        return vectors
