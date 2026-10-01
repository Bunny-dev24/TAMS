"""Cross-turn dependency linker: {entity, modified_at_turn, referenced_at_turns[]}.

Unlike the other extractors, this one has two write modes:
  1. A MODIFICATION of an entity -> new/updated fact, modified_at_turn = this turn.
  2. A REFERENCE to a previously-modified entity -> append this turn to the
     existing fact's referenced_at_turns (no new fact, no overwrite).
"""
from __future__ import annotations

import re
from typing import Any

from tams.extractors.base import BaseExtractor
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore

_MODIFIED_RE = re.compile(
    r"\b(?:renamed|changed|updated|switched|moved|refactored) (?:the |my |our )?"
    r"([A-Za-z0-9_\-\. ]{2,40}?)(?= to\b| from\b| into\b|$)",
    re.IGNORECASE,
)
_ENTITY_CANDIDATE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]{2,30}(?:\.[a-zA-Z0-9_]{2,30})?)\b")


class DependencyExtractor(BaseExtractor):
    category = "dependency"

    def _schema(self) -> dict[str, Any]:
        return {
            "modified": [{"entity": "string"}],
            "referenced": [{"entity": "string"}],
        }

    def _prompt(self, turn: Turn, context_turns: list[Turn]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Identify (a) any named entity (file, parameter, API, module, "
                    "decision topic) that this turn MODIFIES/CHANGES, and (b) any "
                    "named entity this turn merely REFERENCES/mentions without "
                    "changing it. Return JSON {modified: [{entity}], referenced: [{entity}]}."
                ),
            },
            {"role": "user", "content": f"{turn.speaker}: {turn.content}"},
        ]

    def _fake_fn(self, turn: Turn):
        modified = [{"entity": m.group(1).strip()} for m in _MODIFIED_RE.finditer(turn.content)]
        modified_names = {m["entity"].lower() for m in modified}
        referenced = [
            {"entity": e}
            for e in {m.group(1) for m in _ENTITY_CANDIDATE_RE.finditer(turn.content)}
            if e.lower() not in modified_names
        ]
        return {"modified": modified, "referenced": referenced}

    def run(self, turn: Turn, context_turns: list[Turn], fact_store: Tier3FactStore) -> list[dict[str, Any]]:
        try:
            result = self.breaker.call(self._call_llm, turn, context_turns)
        except Exception:
            return []
        try:
            self._persist(fact_store, turn, result)
        except Exception:
            return []
        return (result.get("modified") or []) + (result.get("referenced") or [])

    def _persist(self, fact_store: Tier3FactStore, turn: Turn, extracted: dict[str, Any]) -> None:
        for item in extracted.get("modified", []):
            entity = (item.get("entity") or "").strip()
            if not entity:
                continue
            fact_store.write(
                "dependency",
                turn_id=turn.turn_id,
                confidence=0.8,
                entity=entity,
                modified_at_turn=turn.turn_id,
                referenced_at_turns=[],
            )
        for item in extracted.get("referenced", []):
            entity = (item.get("entity") or "").strip()
            if entity:
                fact_store.record_dependency_reference(entity, turn.turn_id)
