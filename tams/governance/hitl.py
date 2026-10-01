"""Human-in-the-loop correction: lets a user directly fix a wrong stored
fact rather than relying solely on decay/conflict-resolution to silently
sort it out. Produces a labeled, auditable correction trail (useful as
eval data) via FactStore.correct()."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from tams.session import Session
from tams.storage.db import get_session as db_session
from tams.storage.models import CorrectionLog


def correct_fact(session: Session, category: str, fact_id: str, new_fields: dict[str, Any], note: str | None = None) -> str:
    new_id = session.fact_store.fact_store.correct(category, fact_id, new_fields, note=note)
    # Re-embed/re-index the corrected fact so retrieval picks up the fix immediately.
    from tams.tiers.tier3_fact_store import FACT_COLLECTION, _fact_text

    with db_session() as db:
        from tams.storage.fact_db import CATEGORY_MODELS

        model = CATEGORY_MODELS[category]
        row = db.get(model, new_id)
        fields = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    text = _fact_text(category, fields)
    vec = session.llm_client.embed([text], purpose=f"hitl_correction_embed:{category}")[0]
    session.vector_store.upsert(
        FACT_COLLECTION,
        new_id,
        vec,
        {"session_id": session.session_id, "category": category, "fact_id": new_id, "text": text},
    )
    session.bm25.add(session.session_id, new_id, text)
    return new_id


def correction_history(fact_id: str) -> list[dict[str, Any]]:
    with db_session() as db:
        rows = db.execute(select(CorrectionLog).where(CorrectionLog.fact_id == fact_id)).scalars().all()
        return [
            {
                "id": r.id,
                "fact_table": r.fact_table,
                "old_value": r.old_value,
                "new_value": r.new_value,
                "corrected_by": r.corrected_by,
                "note": r.note,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
