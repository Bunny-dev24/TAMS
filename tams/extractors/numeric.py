from __future__ import annotations

import re
from typing import Any

from tams.extractors.base import BaseExtractor
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore

_STOPWORDS = r"is|was|were|the|a|an|and|or|but|to|for|with|set|go|went"
_NUMERIC_RE = re.compile(
    rf"\b(?!(?:{_STOPWORDS})\b)([A-Za-z][A-Za-z0-9_]{{1,30}})\s*(?:=|is set to|set to|is|:)\s*"
    r"(-?\d+(?:\.\d+)?)\s*(ms|s|sec|seconds|%|percent|tokens|turns|px|MB|GB|KB|usd|\$|requests|rpm|days|hours)?\b",
    re.IGNORECASE,
)


class NumericExtractor(BaseExtractor):
    category = "numeric"

    def _schema(self) -> dict[str, Any]:
        return {"items": [{"param": "string", "value": "string", "unit": "string|null"}]}

    def _prompt(self, turn: Turn, context_turns: list[Turn]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Extract every numeric parameter mentioned (thresholds, limits, "
                    "budgets, counts, durations, percentages, prices). Return JSON "
                    "{items: [{param, value, unit}]}. Empty list if none."
                ),
            },
            {"role": "user", "content": f"{turn.speaker}: {turn.content}"},
        ]

    def _fake_fn(self, turn: Turn):
        items = []
        for m in _NUMERIC_RE.finditer(turn.content):
            param, value, unit = m.group(1).strip(), m.group(2), m.group(3)
            items.append({"param": param, "value": value, "unit": unit})
        return {"items": items}

    def _persist(self, fact_store: Tier3FactStore, turn: Turn, extracted: list[dict[str, Any]]) -> None:
        for item in extracted:
            if not item.get("param") or item.get("value") is None:
                continue
            fact_store.write(
                "numeric",
                turn_id=turn.turn_id,
                confidence=0.9,
                param=str(item["param"]).strip(),
                value=str(item["value"]),
                unit=item.get("unit"),
            )
