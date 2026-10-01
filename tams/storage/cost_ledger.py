"""Cost monitoring guardrail: every LLM call (orchestrator, 5 extractors,
reranker, verifier, embeddings) is logged here so cost-per-interaction can be
reported instead of discovered in an invoice."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from tams.config import Settings
from tams.storage.db import get_session
from tams.storage.models import CostLedgerEntry


def make_cost_sink(session_id: str, settings: Settings):
    def sink(purpose: str, model: str, prompt_tokens: int, completion_tokens: int, latency_s: float) -> None:
        cost = (prompt_tokens / 1000.0) * settings.cost_input_per_1k + (
            completion_tokens / 1000.0
        ) * settings.cost_output_per_1k
        with get_session() as db:
            db.add(
                CostLedgerEntry(
                    session_id=session_id,
                    purpose=purpose,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    latency_s=latency_s,
                )
            )

    return sink


def report(session_id: str | None = None, since_hours: float | None = None) -> dict:
    with get_session() as db:
        stmt = select(CostLedgerEntry)
        if session_id:
            stmt = stmt.where(CostLedgerEntry.session_id == session_id)
        if since_hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            stmt = stmt.where(CostLedgerEntry.created_at >= cutoff)
        rows = db.execute(stmt).scalars().all()

    by_purpose: dict[str, dict] = {}
    total_cost = 0.0
    total_calls = 0
    for r in rows:
        bucket = by_purpose.setdefault(
            r.purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0}
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += r.prompt_tokens
        bucket["completion_tokens"] += r.completion_tokens
        bucket["cost_usd"] += r.cost_usd
        bucket["latency_s"] += r.latency_s
        total_cost += r.cost_usd
        total_calls += 1

    for bucket in by_purpose.values():
        bucket["avg_latency_s"] = bucket["latency_s"] / max(bucket["calls"], 1)

    return {
        "total_calls": total_calls,
        "total_cost_usd": total_cost,
        "cost_per_interaction_usd": total_cost / max(total_calls, 1),
        "by_purpose": by_purpose,
    }
