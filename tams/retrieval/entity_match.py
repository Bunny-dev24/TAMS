"""Lightweight entity extraction from a user query, used for the
DETERMINISTIC Tier-3 lookup step (not similarity-based) in the hybrid
retrieval pipeline."""
from __future__ import annotations

import re

_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,40})['\"]")
_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]{2,30}(?:\.[a-zA-Z0-9_]{2,30})?)\b")
_PARAM_LIKE_RE = re.compile(r"\b([a-z][a-z0-9_]{2,30}(?:_[a-z0-9]{2,30}){0,3})\b")


def extract_query_entities(query: str) -> list[str]:
    entities: set[str] = set()
    entities.update(m.group(1) for m in _QUOTED_RE.finditer(query))
    entities.update(m.group(1) for m in _CAPITALIZED_RE.finditer(query))
    entities.update(m.group(1) for m in _PARAM_LIKE_RE.finditer(query) if "_" in m.group(1))
    # Drop very common leading words that are often capitalized at sentence start.
    stop = {"What", "How", "Why", "When", "Where", "Who", "Which", "Can", "Does", "Is", "Are"}
    entities = {e for e in entities if e not in stop}
    return sorted(entities)
