from __future__ import annotations

import re
from typing import Any

from tams.extractors.base import BaseExtractor
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore

_DECISION_RE = re.compile(
    r"(?:we['’]?ll go with|going with|chose|chosen|decided (?:to|on)|opt(?:ed|ing) for)\s+(.+?)"
    r"(?:\s+(?:instead of|over|rather than)\s+(.+?))?(?:\s+because\s+(.+))?[.!]?$",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class DecisionExtractor(BaseExtractor):
    category = "decision"

    def _schema(self) -> dict[str, Any]:
        return {
            "items": [
                {"decision": "string", "rejected_alternatives": "string[]", "reason": "string|null"}
            ]
        }

    def _prompt(self, turn: Turn, context_turns: list[Turn]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Extract explicit decisions made, the alternatives that were "
                    "rejected, and the stated reasoning. Return JSON "
                    "{items: [{decision, rejected_alternatives, reason}]}. Empty list if none."
                ),
            },
            {"role": "user", "content": f"{turn.speaker}: {turn.content}"},
        ]

    def _fake_fn(self, turn: Turn):
        items = []
        for sentence in _split_sentences(turn.content):
            m = _DECISION_RE.search(sentence)
            if m:
                decision = m.group(1).strip().rstrip(".")
                alt = m.group(2)
                reason = m.group(3)
                items.append(
                    {
                        "decision": decision,
                        "rejected_alternatives": [a.strip() for a in (alt or "").split(",") if a.strip()],
                        "reason": reason.strip() if reason else None,
                    }
                )
        return {"items": items}

    def _persist(self, fact_store: Tier3FactStore, turn: Turn, extracted: list[dict[str, Any]]) -> None:
        for item in extracted:
            if not item.get("decision"):
                continue
            fact_store.write(
                "decision",
                turn_id=turn.turn_id,
                confidence=0.85,
                decision=item["decision"],
                rejected_alternatives=item.get("rejected_alternatives") or [],
                reason=item.get("reason"),
            )
