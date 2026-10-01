"""LLM reranker: pools top-2k hybrid candidates down to a final top-k. This
is the step with the largest accuracy payoff per the project's design
philosophy, so it gets a dedicated, swappable component."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tams.config import Settings
from tams.llm.client import LLMClient


@dataclass
class Candidate:
    id: str
    text: str
    source: str  # "tier2" | "tier3"
    prelim_score: float


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class LLMReranker:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        if not candidates:
            return []
        # Cap how many candidates we show the reranker — the "pool" is already
        # capped to retrieval_pool_size upstream, but LLM context has its own limits.
        pool = candidates[: self.settings.retrieval_pool_size]

        numbered = "\n".join(f"[{i}] ({c.source}) {c.text[:400]}" for i, c in enumerate(pool))
        schema = {"ranked_ids": "int[]"}
        messages = [
            {
                "role": "system",
                "content": (
                    f"Given the query, rank the {len(pool)} candidate memory items "
                    f"by relevance. Return JSON {{ranked_ids: [indices]}} listing the "
                    f"top {top_k} indices, most relevant first."
                ),
            },
            {"role": "user", "content": f"QUERY: {query}\n\nCANDIDATES:\n{numbered}"},
        ]

        def _fake_rerank(_msgs):
            q_tokens = _tokenize(query)
            scored = []
            for i, c in enumerate(pool):
                overlap = len(q_tokens & _tokenize(c.text))
                scored.append((i, overlap + c.prelim_score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return {"ranked_ids": [i for i, _ in scored[:top_k]]}

        result = self.llm_client.chat_json(
            messages,
            model=self.settings.reranker_model,
            schema=schema,
            purpose="reranker",
            fake_fn=_fake_rerank,
        )
        ids = result.get("ranked_ids", []) if isinstance(result, dict) else []
        ranked = []
        for i in ids:
            if isinstance(i, int) and 0 <= i < len(pool):
                ranked.append(pool[i])
        # Fallback: if the LLM returned garbage, keep prelim ordering.
        if not ranked:
            ranked = sorted(pool, key=lambda c: c.prelim_score, reverse=True)[:top_k]
        return ranked[:top_k]
