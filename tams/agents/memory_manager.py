"""Memory Manager Agent — runs ASYNC (never in the user-facing hot path):

  * fires the 5 extractors (4 per-turn + 1 periodic-batch) using the cheap
    mini-tier model
  * maintains the anchored, extend-only session-state summary
  * triggers Tier 1 -> Tier 2 compaction when the working-memory budget is hit
  * conflict resolution happens inside the Tier-3 fact store writes
    themselves (timestamp + confidence based, store-both-never-overwrite)
"""
from __future__ import annotations

from tams.extractors import PER_TURN_EXTRACTORS, PreferenceDetector
from tams.session import Session
from tams.storage.working_memory import Turn


class MemoryManager:
    def __init__(self, session: Session):
        self.session = session
        self.settings = session.settings
        self.extractors = [cls(session.llm_client, session.settings) for cls in PER_TURN_EXTRACTORS]
        self.preference_detector = PreferenceDetector(session.llm_client, session.settings)

    def process_turn(self, turn: Turn, context_turns: list[Turn]) -> dict:
        """Synchronous core logic. Callers (Orchestrator) dispatch this onto
        the shared background ThreadPoolExecutor so it never blocks the
        response path."""
        report: dict = {"extracted": {}, "compacted": False, "preference_run": False}

        for extractor in self.extractors:
            try:
                items = extractor.run(turn, context_turns, self.session.fact_store)
                report["extracted"][extractor.category] = len(items)
            except Exception as e:  # per guardrail: never let extraction crash the pipeline
                report["extracted"][extractor.category] = f"error: {e}"

        self.session.turns_since_last_preference_run += 1
        if self.preference_detector.should_run(self.session.turns_since_last_preference_run):
            recent = self.session.working_memory.get_all(self.session.session_id)
            window = recent[-self.settings.preference_batch_interval_turns :]
            try:
                items = self.preference_detector.run_batch(window, self.session.fact_store)
                report["preference_run"] = True
                report["preference_items"] = len(items)
            except Exception as e:
                report["preference_run_error"] = str(e)
            self.session.turns_since_last_preference_run = 0

        if self.session.working_memory.needs_compaction(self.session.session_id):
            evicted = self.session.working_memory.evict_for_compaction(self.session.session_id)
            if evicted:
                chunks = self.session.chunk_store.ingest_evicted_turns(self.session.session_id, evicted)
                self.session.chunk_store.extend_summary(self.session.session_id, chunks, user_id=self.session.user_id)
                report["compacted"] = True
                report["evicted_turns"] = len(evicted)
                report["chunks_created"] = len(chunks)

        return report

    def dispatch_async(self, turn: Turn, context_turns: list[Turn]):
        """Fire-and-forget onto the shared executor — satisfies guardrail
        'no recursive LLM calls in user-facing hot path, only async
        background'. Returns the Future so callers (and tests) *may* await
        it, but the orchestrator itself never blocks on it."""
        return self.session.executor.submit(self.process_turn, turn, context_turns)