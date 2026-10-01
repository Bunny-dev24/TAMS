"""Tier 3: Structured Fact Store — CRUD + conflict resolution + decay-aware
retrieval + privacy export/delete, across the 5 protected categories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from tams.decay import decay_weight
from tams.storage.db import get_session
from tams.storage.models import (
    ConstraintFact,
    CorrectionLog,
    DecisionFact,
    DependencyFact,
    NumericFact,
    PreferenceFact,
)

CATEGORY_MODELS = {
    "numeric": NumericFact,
    "constraint": ConstraintFact,
    "decision": DecisionFact,
    "dependency": DependencyFact,
    "preference": PreferenceFact,
}

# The field(s) used to detect "this is the same logical fact, possibly updated"
_CONFLICT_KEY_FIELD = {
    "numeric": "param",
    "constraint": "scope",
    "decision": None,  # decisions are additive by nature; no auto-conflict key
    "dependency": "entity",
    "preference": "pref_type",
}


@dataclass
class FactWriteResult:
    fact_id: str
    conflicted_with: str | None = None


class FactStore:
    def __init__(self, session_id: str, user_id: str = "default"):
        self.session_id = session_id
        self.user_id = user_id

    # --------------------------------------------------------------- write
    def write(self, category: str, turn_id: int, confidence: float, **fields) -> FactWriteResult:
        model = CATEGORY_MODELS[category]
        key_field = _CONFLICT_KEY_FIELD[category]
        conflicted_with = None
        with get_session() as db:
            if key_field and fields.get(key_field):
                existing = db.execute(
                    select(model).where(
                        model.session_id == self.session_id,
                        model.status == "active",
                        getattr(model, key_field) == fields[key_field],
                    )
                ).scalars().all()
                new_value = fields.get("value") or fields.get("constraint_text") or fields.get("decision")
                for row in existing:
                    # Conflict iff the new value materially differs (string compare
                    # is deliberately simple/deterministic here).
                    old_value = getattr(row, "value", None) or getattr(row, "constraint_text", None) or getattr(
                        row, "decision", None
                    )
                    if new_value is not None and old_value is not None and str(new_value) == str(old_value):
                        # Idempotent restatement of the SAME fact — not a new
                        # fact and not a conflict. Just bump usage/recency
                        # rather than spawning a duplicate active row.
                        row.usage_count += 1
                        row.last_used_turn = turn_id
                        return FactWriteResult(fact_id=row.id, conflicted_with=None)
                    if new_value is not None and old_value is not None and str(new_value) != str(old_value):
                        row.status = "superseded"
                        conflicted_with = row.id
                        # old row kept forever for audit — never silently overwritten.

            row = model(
                session_id=self.session_id,
                user_id=self.user_id,
                turn_id=turn_id,
                confidence=confidence,
                last_used_turn=turn_id,
                **fields,
            )
            db.add(row)
            db.flush()
            fact_id = row.id
            if conflicted_with:
                row2 = db.get(model, conflicted_with)
                if row2:
                    row2.superseded_by = fact_id
        return FactWriteResult(fact_id=fact_id, conflicted_with=conflicted_with)

    # ------------------------------------------------------------ querying
    def active_facts(self, category: str, current_turn: int | None = None) -> list[dict[str, Any]]:
        model = CATEGORY_MODELS[category]
        with get_session() as db:
            rows = db.execute(
                select(model).where(model.session_id == self.session_id, model.status == "active")
            ).scalars().all()
            return [self._row_to_dict(r, category) for r in rows]

    def match_entities(self, category: str, entities: list[str]) -> list[dict[str, Any]]:
        """Deterministic lookup used by the retrieval pipeline: 'any Tier 3
        facts matching query entities' regardless of similarity score."""
        if not entities:
            return []
        lowered = {e.lower() for e in entities}
        results = []
        for fact in self.active_facts(category):
            haystack = " ".join(str(v) for v in fact.values()).lower()
            if any(e in haystack for e in lowered):
                results.append(fact)
        return results

    def all_active_across_categories(self) -> list[dict[str, Any]]:
        out = []
        for cat in CATEGORY_MODELS:
            out.extend(self.active_facts(cat))
        return out

    def ranked_by_decay(self, category: str, current_turn: int, half_life_turns: float) -> list[dict[str, Any]]:
        facts = self.active_facts(category)
        for f in facts:
            f["_decay_weight"] = decay_weight(
                f.get("last_used_turn") or f.get("turn_id") or 0,
                current_turn,
                f.get("usage_count", 0),
                half_life_turns,
            )
        facts.sort(key=lambda f: f["_decay_weight"], reverse=True)
        return facts

    def mark_used(self, category: str, fact_id: str, turn_id: int) -> None:
        model = CATEGORY_MODELS[category]
        with get_session() as db:
            row = db.get(model, fact_id)
            if row:
                row.usage_count += 1
                row.last_used_turn = turn_id

    def record_dependency_reference(self, entity: str, turn_id: int) -> list[str]:
        """Cross-turn dependency linker helper: when `entity` (previously
        modified at some turn) is referenced again at `turn_id`, append the
        reference without touching modified_at_turn. Fuzzy substring match
        keeps this robust to minor phrasing differences."""
        updated = []
        with get_session() as db:
            rows = db.execute(
                select(DependencyFact).where(
                    DependencyFact.session_id == self.session_id, DependencyFact.status == "active"
                )
            ).scalars().all()
            el = entity.lower().strip()
            for row in rows:
                rl = row.entity.lower().strip()
                if el and rl and (el in rl or rl in el):
                    refs = list(row.referenced_at_turns or [])
                    if turn_id not in refs:
                        refs.append(turn_id)
                        row.referenced_at_turns = refs
                    updated.append(row.id)
        return updated

    # ------------------------------------------------------- HITL correction
    def correct(self, category: str, fact_id: str, new_fields: dict[str, Any], note: str | None = None) -> str:
        """Direct user correction of a stored fact. The old row is marked
        superseded (never silently discarded); a correction log entry is
        written for audit + future eval labeling."""
        model = CATEGORY_MODELS[category]
        with get_session() as db:
            row = db.get(model, fact_id)
            if row is None:
                raise ValueError(f"fact {fact_id} not found in {category}")
            old_value = self._row_to_dict(row, category)
            row.status = "superseded"
            new_row = model(
                session_id=row.session_id,
                user_id=row.user_id,
                turn_id=row.turn_id,
                confidence=1.0,  # user correction -> max confidence
                last_used_turn=row.last_used_turn,
            )
            base = self._row_to_dict(row, category)
            base.update(new_fields)
            for k, v in base.items():
                if hasattr(new_row, k) and k not in ("id", "status", "created_at"):
                    setattr(new_row, k, v)
            db.add(new_row)
            db.flush()
            row.superseded_by = new_row.id
            db.add(
                CorrectionLog(
                    fact_table=category,
                    fact_id=fact_id,
                    old_value=old_value,
                    new_value=self._row_to_dict(new_row, category),
                    corrected_by="user",
                    note=note,
                )
            )
            return new_row.id

    # --------------------------------------------------------------- privacy
    def export_all(self) -> dict[str, Any]:
        """GDPR-style data export for this session/user (all categories, all
        statuses, including superseded, for full transparency)."""
        out: dict[str, Any] = {}
        for cat, model in CATEGORY_MODELS.items():
            with get_session() as db:
                rows = db.execute(select(model).where(model.session_id == self.session_id)).scalars().all()
                out[cat] = [self._row_to_dict(r, cat) for r in rows]
        return out

    def delete_all(self) -> int:
        """Right-to-forget: hard delete every fact for this session/user
        (the one place we DO hard-delete — explicit user-initiated erasure)."""
        count = 0
        for cat, model in CATEGORY_MODELS.items():
            with get_session() as db:
                rows = db.execute(select(model).where(model.session_id == self.session_id)).scalars().all()
                for r in rows:
                    db.delete(r)
                    count += 1
        return count

    @staticmethod
    def _row_to_dict(row, category: str) -> dict[str, Any]:
        d = {}
        for c in row.__table__.columns:
            v = getattr(row, c.name)
            d[c.name] = v.isoformat() if hasattr(v, "isoformat") else v
        d["category"] = category
        return d
