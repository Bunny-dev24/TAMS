"""Regression test: a failed Tier-2 write (e.g. an embedding-API error) must
NEVER silently lose raw Tier-1 turns. This directly protects the project's
'lossless memory' guarantee against exactly the kind of failure a flaky or
incompatible LLM/embedding provider could trigger."""
from __future__ import annotations

import pytest

from tams.agents.memory_manager import MemoryManager
from tams.agents.orchestrator import OrchestratorAgent
from tams.session import build_session
from tams.storage.working_memory import Turn


def test_plan_eviction_is_non_destructive(settings):
    session = build_session("safe-compact-1", settings)
    for i in range(1, 11):
        session.working_memory.append(
            Turn(turn_id=i, session_id=session.session_id, speaker="user", content="x" * 200)
        )

    before = session.working_memory.get_all(session.session_id)
    planned = session.working_memory.plan_eviction(session.session_id, target_fraction=0.0)
    after = session.working_memory.get_all(session.session_id)

    assert len(planned) > 0
    assert after == before  # nothing removed by planning alone


def test_failed_tier2_ingestion_does_not_lose_tier1_turns(settings, monkeypatch):
    settings = settings.model_copy(update={"working_memory_max_turns": 5})
    session = build_session("safe-compact-2", settings)
    orch = OrchestratorAgent(session, async_memory=False)
    mm = MemoryManager(session)

    for i in range(1, 11):
        session.working_memory.append(
            Turn(turn_id=i, session_id=session.session_id, speaker="user", content="x" * 200)
        )
    turns_before = session.working_memory.get_all(session.session_id)
    assert len(turns_before) == 10

    def _boom(*args, **kwargs):
        raise RuntimeError("embedding API does not support this provider")

    monkeypatch.setattr(session.chunk_store, "ingest_evicted_turns", _boom)

    report = mm.process_turn(
        Turn(turn_id=11, session_id=session.session_id, speaker="user", content="trigger"),
        turns_before,
    )

    assert "compaction_error" in report
    assert report["compacted"] is False
    # The critical assertion: every turn that existed before the failed
    # compaction attempt must STILL be there — nothing was popped.
    turns_after = session.working_memory.get_all(session.session_id)
    before_ids = {t.turn_id for t in turns_before}
    after_ids = {t.turn_id for t in turns_after}
    assert before_ids.issubset(after_ids)


def test_successful_compaction_still_removes_turns_from_tier1(settings):
    settings = settings.model_copy(update={"working_memory_max_turns": 5})
    session = build_session("safe-compact-3", settings)
    mm = MemoryManager(session)

    for i in range(1, 11):
        session.working_memory.append(
            Turn(turn_id=i, session_id=session.session_id, speaker="user", content="x" * 200)
        )
    turns_before = session.working_memory.get_all(session.session_id)

    report = mm.process_turn(
        Turn(turn_id=11, session_id=session.session_id, speaker="user", content="trigger"),
        turns_before,
    )

    assert report["compacted"] is True
    turns_after = session.working_memory.get_all(session.session_id)
    assert len(turns_after) < len(turns_before)
