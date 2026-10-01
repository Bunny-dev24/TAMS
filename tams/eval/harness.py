"""Evaluation harness tying together: synthetic needle-in-haystack data,
the real TAMS pipeline (or a baseline), and the metrics module.

Primary measurement: recall@k PER PROTECTED CATEGORY (numeric, constraint,
decision, dependency, preference) — not just generic fact recall.
"""
from __future__ import annotations

import uuid

from tams.agents.memory_manager import MemoryManager
from tams.agents.orchestrator import OrchestratorAgent
from tams.config import Settings
from tams.eval.metrics import EvalReport, aggregate_results, check_hit, compaction_ratio
from tams.eval.synth_data import HaystackSpec
from tams.session import build_session, reset_shared_state
from tams.storage.working_memory import Turn
from tams.tokens import count_tokens


def _ingest_haystack(
    orchestrator: OrchestratorAgent,
    memory_manager: MemoryManager,
    haystack: HaystackSpec,
    enable_extractors: bool,
) -> int:
    session = orchestrator.session
    raw_tokens = 0
    for speaker, content in haystack.turns:
        session.turn_counter += 1
        turn = Turn(turn_id=session.turn_counter, session_id=session.session_id, speaker=speaker, content=content)
        session.working_memory.append(turn)
        raw_tokens += count_tokens(content)

        context_turns = session.working_memory.get_all(session.session_id)[-8:]
        if enable_extractors:
            memory_manager.process_turn(turn, context_turns)
        else:
            # Still allow Tier1->Tier2 compaction (raw chunks / anchored
            # summary), just skip the 5 dedicated extractors — this is the
            # "without dedicated extractors" ablation.
            if session.working_memory.needs_compaction(session.session_id):
                evicted = session.working_memory.evict_for_compaction(session.session_id)
                if evicted:
                    chunks = session.chunk_store.ingest_evicted_turns(session.session_id, evicted)
                    session.chunk_store.extend_summary(session.session_id, chunks, user_id=session.user_id)
    return raw_tokens


def run_tams_eval(
    haystack: HaystackSpec,
    settings: Settings,
    enable_hybrid_rerank: bool = True,
    enable_extractors: bool = True,
    enable_decay: bool = True,
    session_id: str | None = None,
) -> EvalReport:
    reset_shared_state()
    session_id = session_id or f"eval-{uuid.uuid4().hex[:8]}"
    session = build_session(session_id, settings)
    session.retriever.enable_hybrid_rerank = enable_hybrid_rerank

    orchestrator = OrchestratorAgent(session, async_memory=False)
    memory_manager = MemoryManager(session)

    raw_tokens = _ingest_haystack(orchestrator, memory_manager, haystack, enable_extractors)
    # Measure the post-ingestion Tier-1 footprint BEFORE the query phase adds
    # its own turns, so the ratio reflects compaction of the haystack itself.
    post_ingest_tier1_tokens = sum(
        count_tokens(t.content) for t in session.working_memory.get_all(session.session_id)
    )

    per_query_results = []
    for needle in haystack.needles:
        result = orchestrator.handle_turn(needle.query)
        # Patch decay ablation onto this one retrieval if requested, by
        # re-running retrieval directly (handle_turn already consumed one
        # turn id, which is fine/expected for an eval conversation).
        if not enable_decay:
            retrieval = session.retriever.retrieve(
                session.session_id, needle.query, session.turn_counter, session.fact_store, enable_decay=False
            )
        else:
            retrieval = result.retrieval

        hit = check_hit(needle.expected_answer_fragment, retrieval.context_text)
        per_query_results.append(
            {
                "category": needle.category,
                "hit": hit,
                "latency_s": retrieval.latency_s,
                "hallucination_rate": result.verification.hallucination_rate,
                "citation_coverage": result.verification.citation_coverage,
            }
        )

    report = aggregate_results(per_query_results)
    report.compaction_token_reduction_ratio = compaction_ratio(raw_tokens, post_ingest_tier1_tokens)
    return report


def run_full_context_baseline(haystack: HaystackSpec, settings: Settings) -> EvalReport:
    """Full-context stuffing: every needle's answer is trivially 'in context'
    by construction (nothing is ever evicted) — included as the recall
    upper bound / cost-and-latency lower-bound-breaker reference point."""
    from tams.eval.baselines import full_context_stuffing

    turns = []
    for i, (speaker, content) in enumerate(haystack.turns, start=1):
        turns.append(Turn(turn_id=i, session_id="baseline", speaker=speaker, content=content))
    context_text = full_context_stuffing(turns)

    per_query_results = [
        {"category": n.category, "hit": check_hit(n.expected_answer_fragment, context_text), "latency_s": 0.0}
        for n in haystack.needles
    ]
    return aggregate_results(per_query_results)
