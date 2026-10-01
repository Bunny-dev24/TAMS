"""Implicit preference detector: periodic BATCH job (every ~20 turns, not
per-turn) — {pref_type, value, confidence, evidence_turns[]}.

Cheaper than running per-turn since implicit preferences (tone, verbosity,
tooling choices, style) are only reliably inferred from a window of turns,
not a single message.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tams.config import Settings
from tams.llm.circuit_breaker import CircuitOpenError, get_breaker
from tams.llm.client import LLMClient
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore

_VERBOSITY_SHORT = re.compile(r"\b(briefly|short answer|tl;?dr|just the code|no explanation)\b", re.IGNORECASE)
_VERBOSITY_LONG = re.compile(r"\b(explain in detail|walk me through|be thorough|step by step)\b", re.IGNORECASE)
_TOOL_MENTION = re.compile(r"\b(pytest|docker|redis|qdrant|typescript|python|react|postgres)\b", re.IGNORECASE)


class PreferenceDetector:
    category = "preference"

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings
        self.breaker = get_breaker(
            "extractor:preference",
            failure_threshold=settings.extractor_circuit_breaker_threshold,
            cooldown_s=settings.extractor_circuit_breaker_cooldown_s,
        )

    def should_run(self, turns_since_last_run: int) -> bool:
        return turns_since_last_run >= self.settings.preference_batch_interval_turns

    def run_batch(self, turns: list[Turn], fact_store: Tier3FactStore) -> list[dict[str, Any]]:
        try:
            result = self.breaker.call(self._call_llm, turns)
        except CircuitOpenError:
            return []
        except Exception:
            return []
        items = result.get("items", []) if isinstance(result, dict) else result
        for item in items:
            if not item.get("pref_type") or item.get("value") is None:
                continue
            try:
                fact_store.write(
                    "preference",
                    turn_id=turns[-1].turn_id if turns else 0,
                    confidence=float(item.get("confidence", 0.6)),
                    pref_type=item["pref_type"],
                    value=str(item["value"]),
                    evidence_turns=item.get("evidence_turns", []),
                )
            except Exception:
                continue
        return items

    def _schema(self) -> dict[str, Any]:
        return {"items": [{"pref_type": "string", "value": "string", "confidence": "number", "evidence_turns": "int[]"}]}

    def _call_llm(self, turns: list[Turn]):
        transcript = "\n".join(f"[turn {t.turn_id}] {t.speaker}: {t.content}" for t in turns)
        messages = [
            {
                "role": "system",
                "content": (
                    "Infer IMPLICIT user preferences from this window of conversation "
                    "(communication style, verbosity, tooling choices, formatting "
                    "preferences). Only report patterns with real evidence across "
                    "multiple turns. Return JSON {items: [{pref_type, value, "
                    "confidence, evidence_turns}]}."
                ),
            },
            {"role": "user", "content": transcript},
        ]
        return self.llm_client.chat_json(
            messages,
            model=self.settings.extractor_model,
            schema=self._schema(),
            purpose="extractor:preference",
            fake_fn=lambda _msgs: self._fake_fn(turns),
        )

    def _fake_fn(self, turns: list[Turn]) -> dict[str, Any]:
        items = []
        short_turns = [t.turn_id for t in turns if _VERBOSITY_SHORT.search(t.content)]
        long_turns = [t.turn_id for t in turns if _VERBOSITY_LONG.search(t.content)]
        if len(short_turns) >= 2:
            items.append(
                {"pref_type": "verbosity", "value": "concise", "confidence": 0.7, "evidence_turns": short_turns}
            )
        if len(long_turns) >= 2:
            items.append(
                {"pref_type": "verbosity", "value": "detailed", "confidence": 0.7, "evidence_turns": long_turns}
            )
        tool_counts = Counter()
        tool_turns: dict[str, list[int]] = {}
        for t in turns:
            for m in _TOOL_MENTION.finditer(t.content):
                tool = m.group(1).lower()
                tool_counts[tool] += 1
                tool_turns.setdefault(tool, []).append(t.turn_id)
        for tool, count in tool_counts.items():
            if count >= 3:
                items.append(
                    {
                        "pref_type": "preferred_tool",
                        "value": tool,
                        "confidence": min(0.5 + 0.1 * count, 0.95),
                        "evidence_turns": tool_turns[tool],
                    }
                )
        return {"items": items}
