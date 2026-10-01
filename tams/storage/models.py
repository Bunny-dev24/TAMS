"""SQLAlchemy ORM models for Tier 3 (structured facts) + supporting tables:
anchored session summaries, cost ledger, HITL correction log, and the
cross-session user profile store."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from tams.storage.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FactMixin:
    """Shared bookkeeping for every Tier-3 fact category.

    status: active | superseded | deleted  (never hard-deleted on conflict —
    superseded facts are kept for audit; `deleted` is reserved for explicit
    privacy/right-to-forget erasure.)
    """

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True, default="default")
    turn_id: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String, default="active")  # active|superseded|deleted
    superseded_by: Mapped[str | None] = mapped_column(String, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_turn: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class NumericFact(FactMixin, Base):
    __tablename__ = "numeric_facts"
    param: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[str] = mapped_column(String)  # stored as str, caller coerces type
    unit: Mapped[str | None] = mapped_column(String, nullable=True)


class ConstraintFact(FactMixin, Base):
    __tablename__ = "constraint_facts"
    constraint_text: Mapped[str] = mapped_column(Text)
    scope: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class DecisionFact(FactMixin, Base):
    __tablename__ = "decision_facts"
    decision: Mapped[str] = mapped_column(Text)
    rejected_alternatives: Mapped[list] = mapped_column(JSON, default=list)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class DependencyFact(FactMixin, Base):
    __tablename__ = "dependency_facts"
    entity: Mapped[str] = mapped_column(String, index=True)
    modified_at_turn: Mapped[int] = mapped_column(Integer)
    referenced_at_turns: Mapped[list] = mapped_column(JSON, default=list)


class PreferenceFact(FactMixin, Base):
    __tablename__ = "preference_facts"
    pref_type: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[str] = mapped_column(String)
    evidence_turns: Mapped[list] = mapped_column(JSON, default=list)


class SessionSummary(Base):
    """Anchored, extend-only running session-state document (Tier 2 lightweight
    summary). Never regenerated from scratch — only appended to per eviction
    span, per spec."""

    __tablename__ = "session_summaries"
    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True, default="default")
    intent: Mapped[str] = mapped_column(Text, default="")
    changes_made: Mapped[list] = mapped_column(JSON, default=list)
    decisions: Mapped[list] = mapped_column(JSON, default=list)
    next_steps: Mapped[list] = mapped_column(JSON, default=list)
    last_anchored_turn: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class CostLedgerEntry(Base):
    __tablename__ = "cost_ledger"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    purpose: Mapped[str] = mapped_column(String, index=True)  # orchestrator|extractor:numeric|verifier|...
    model: Mapped[str] = mapped_column(String)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_s: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class CorrectionLog(Base):
    """Human-in-the-loop corrections — users directly fix a wrong stored fact.
    Kept as labeled audit trail / eval data, never silently discarded."""

    __tablename__ = "correction_log"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    fact_table: Mapped[str] = mapped_column(String)
    fact_id: Mapped[str] = mapped_column(String, index=True)
    old_value: Mapped[dict] = mapped_column(JSON)
    new_value: Mapped[dict] = mapped_column(JSON)
    corrected_by: Mapped[str] = mapped_column(String, default="user")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UserProfile(Base):
    """Multi-session identity/continuity: durable, user-level facts that
    persist across conversations, separate from per-session Tier 3 memory."""

    __tablename__ = "user_profile"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, index=True)
    pref_type: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
