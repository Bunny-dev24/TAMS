from __future__ import annotations

import re
from typing import Any

from tams.extractors.base import BaseExtractor
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore

_CONSTRAINT_MARKERS = re.compile(
    r"\b(must(?: not)?|should(?: not)?|never|always|cannot|can't|required to|not allowed to|has to|forbidden)\b",
    re.IGNORECASE,
)
_REASON_RE = re.compile(r"\bbecause\b(.*)", re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class ConstraintExtractor(BaseExtractor):
    category = "constraint"

    def _schema(self) -> dict[str, Any]:
        return {"items": [{"constraint_text": "string", "scope": "string", "reason": "string|null"}]}

    def _prompt(self, turn: Turn, context_turns: list[Turn]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Extract hard constraints/requirements/rules stated by the user "
                    "(things that MUST/MUST NOT happen). Return JSON "
                    "{items: [{constraint_text, scope, reason}]}. Empty list if none."
                ),
            },
            {"role": "user", "content": f"{turn.speaker}: {turn.content}"},
        ]

    def _fake_fn(self, turn: Turn):
        items = []
        for sentence in _split_sentences(turn.content):
            if _CONSTRAINT_MARKERS.search(sentence):
                reason_match = _REASON_RE.search(sentence)
                items.append(
                    {
                        "constraint_text": sentence,
                        "scope": "global",
                        "reason": reason_match.group(1).strip() if reason_match else None,
                    }
                )
        return {"items": items}

    def _persist(self, fact_store: Tier3FactStore, turn: Turn, extracted: list[dict[str, Any]]) -> None:
        for item in extracted:
            if not item.get("constraint_text"):
                continue
            fact_store.write(
                "constraint",
                turn_id=turn.turn_id,
                confidence=0.85,
                constraint_text=item["constraint_text"],
                scope=item.get("scope") or "global",
                reason=item.get("reason"),
            )
