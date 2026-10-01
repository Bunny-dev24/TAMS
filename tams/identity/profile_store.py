"""Multi-session identity/continuity: a user-level persistent profile store,
separate from per-conversation (Tier 1/2/3) memory. Durable preferences
detected within a session can be PROMOTED here so they carry across
sessions, instead of being re-learned from scratch every conversation."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from tams.storage.db import get_session
from tams.storage.models import UserProfile

PROMOTION_CONFIDENCE_THRESHOLD = 0.75


class UserProfileStore:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def upsert(self, pref_type: str, value: str, confidence: float, source_session_id: str | None = None) -> str:
        with get_session() as db:
            existing = db.execute(
                select(UserProfile).where(
                    UserProfile.user_id == self.user_id,
                    UserProfile.pref_type == pref_type,
                    UserProfile.status == "active",
                )
            ).scalars().all()
            for row in existing:
                if str(row.value) != str(value):
                    row.status = "superseded"
            row = UserProfile(
                user_id=self.user_id,
                pref_type=pref_type,
                value=str(value),
                confidence=confidence,
                source_session_id=source_session_id,
            )
            db.add(row)
            db.flush()
            return row.id

    def get_active(self) -> list[dict[str, Any]]:
        with get_session() as db:
            rows = db.execute(
                select(UserProfile).where(UserProfile.user_id == self.user_id, UserProfile.status == "active")
            ).scalars().all()
            return [
                {
                    "id": r.id,
                    "pref_type": r.pref_type,
                    "value": r.value,
                    "confidence": r.confidence,
                    "source_session_id": r.source_session_id,
                }
                for r in rows
            ]

    def promote_from_session_facts(self, session_preference_facts: list[dict[str, Any]], session_id: str) -> int:
        """Called periodically (e.g. alongside the preference-detector batch
        run) to lift high-confidence, session-scoped preferences into the
        durable cross-session profile."""
        promoted = 0
        for fact in session_preference_facts:
            if fact.get("confidence", 0) >= PROMOTION_CONFIDENCE_THRESHOLD:
                self.upsert(fact["pref_type"], fact["value"], fact["confidence"], source_session_id=session_id)
                promoted += 1
        return promoted

    def delete_all(self) -> int:
        with get_session() as db:
            rows = db.execute(select(UserProfile).where(UserProfile.user_id == self.user_id)).scalars().all()
            for r in rows:
                db.delete(r)
            return len(rows)
