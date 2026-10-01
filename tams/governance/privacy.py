"""Privacy / data-governance guardrail: GDPR-style right-to-export and
right-to-forget, spanning every tier (Tier 1 raw turns, Tier 2 chunks +
anchored summary, Tier 3 structured facts, cross-session user profile)."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from tams.config import Settings
from tams.identity.profile_store import UserProfileStore
from tams.session import Session, build_session
from tams.storage.db import get_session as db_session


def export_session(session: Session) -> dict[str, Any]:
    tier1 = [t.to_dict() for t in session.working_memory.get_all(session.session_id)]
    tier2_hits = session.vector_store.get_all(
        "tier2_chunks", filter={"session_id": session.session_id}
    )
    tier2 = [h.payload for h in tier2_hits]
    tier3 = session.fact_store.fact_store.export_all()
    summary = session.chunk_store.get_summary(session.session_id)
    summary_dict = (
        {
            "intent": summary.intent,
            "changes_made": summary.changes_made,
            "decisions": summary.decisions,
            "next_steps": summary.next_steps,
            "last_anchored_turn": summary.last_anchored_turn,
        }
        if summary
        else None
    )
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "tier1_working_memory": tier1,
        "tier2_chunks": tier2,
        "tier3_facts": tier3,
        "anchored_summary": summary_dict,
    }


def delete_session(session: Session) -> dict[str, int]:
    """Right-to-forget for a single session. This is the one place TAMS does
    a true hard delete — every other write path is append-only/supersede."""
    deleted_facts = session.fact_store.fact_store.delete_all()
    session.chunk_store.delete_session(session.session_id)
    # Tier 1 (working memory): drop remaining raw turns.
    remaining = session.working_memory.get_all(session.session_id)
    session.working_memory.store.pop_oldest(session.session_id, len(remaining))
    return {"facts_deleted": deleted_facts, "tier1_turns_deleted": len(remaining)}


def find_session_ids_for_user(user_id: str) -> set[str]:
    from tams.storage.models import (
        ConstraintFact,
        DecisionFact,
        DependencyFact,
        NumericFact,
        PreferenceFact,
        SessionSummary,
    )

    session_ids: set[str] = set()
    with db_session() as db:
        for model in (NumericFact, ConstraintFact, DecisionFact, DependencyFact, PreferenceFact, SessionSummary):
            rows = db.execute(select(model.session_id).where(model.user_id == user_id)).scalars().all()
            session_ids.update(rows)
    return session_ids


def export_user(user_id: str, settings: Settings) -> dict[str, Any]:
    sessions_data = []
    for sid in find_session_ids_for_user(user_id):
        session = build_session(sid, settings, user_id=user_id)
        sessions_data.append(export_session(session))
    profile = UserProfileStore(user_id).get_active()
    return {"user_id": user_id, "sessions": sessions_data, "profile": profile}


def delete_user(user_id: str, settings: Settings) -> dict[str, Any]:
    results = []
    for sid in find_session_ids_for_user(user_id):
        session = build_session(sid, settings, user_id=user_id)
        results.append({"session_id": sid, **delete_session(session)})
    profile_deleted = UserProfileStore(user_id).delete_all()
    return {"user_id": user_id, "sessions": results, "profile_rows_deleted": profile_deleted}
