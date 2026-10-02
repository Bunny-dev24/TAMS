"""Tier 1 — Working Memory (Hot): last N turns, raw, unmodified.

Fixed token budget = working_memory_budget_fraction * context_window_tokens.
Compaction triggers at that fraction (<=70% per spec, default 65%), evicting
the OLDEST turns down to a safe low-water mark, handing them to Tier 2.
"""
from __future__ import annotations

from tams.config import Settings
from tams.storage.working_memory import Turn, WorkingMemoryStore
from tams.tokens import count_tokens


class WorkingMemory:
    def __init__(self, store: WorkingMemoryStore, settings: Settings):
        self.store = store
        self.settings = settings

    def _count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def budget_tokens(self) -> int:
        return int(self.settings.context_window_tokens * self.settings.working_memory_budget_fraction)

    def current_tokens(self, session_id: str) -> int:
        return sum(self._count_tokens(t.content) for t in self.store.get_all(session_id))

    def append(self, turn: Turn) -> None:
        self.store.append(turn)

    def get_all(self, session_id: str) -> list[Turn]:
        return self.store.get_all(session_id)

    def needs_compaction(self, session_id: str) -> bool:
        turns = self.store.get_all(session_id)
        if len(turns) > self.settings.working_memory_max_turns:
            return True
        return self.current_tokens(session_id) >= self.budget_tokens()

    def plan_eviction(self, session_id: str, target_fraction: float = 0.5) -> list[Turn]:
        """Non-destructively determine which oldest turns SHOULD be evicted
        to bring Tier 1 back under budget — nothing is removed from storage
        here. Callers MUST call commit_eviction() with the same count, and
        ONLY after those turns have been durably persisted into Tier 2 — this
        is what prevents a failed Tier-2 write (e.g. an embedding-API error)
        from silently losing raw turns that were already popped out of Tier 1.
        """
        target_tokens = int(self.budget_tokens() * target_fraction)
        all_turns = self.store.get_all(session_id)
        n = len(all_turns)
        running_tokens = sum(self._count_tokens(t.content) for t in all_turns)
        i = 0
        while i < n and (running_tokens > target_tokens or (n - i) > self.settings.working_memory_max_turns):
            running_tokens -= self._count_tokens(all_turns[i].content)
            i += 1
        # Round up to a whole number of Tier-2 chunk groups for clean boundaries.
        chunk_size = max(self.settings.chunk_size_turns, 1)
        i = min(((i + chunk_size - 1) // chunk_size) * chunk_size, n)
        return all_turns[:i]

    def commit_eviction(self, session_id: str, count: int) -> None:
        """Actually remove the first `count` turns from Tier 1. Call this
        ONLY after the same turns have been safely written to Tier 2."""
        if count > 0:
            self.store.pop_oldest(session_id, count)

    def evict_for_compaction(self, session_id: str, target_fraction: float = 0.5) -> list[Turn]:
        """Convenience wrapper that plans AND immediately commits eviction
        (not failure-safe — if the caller's subsequent Tier-2 write fails,
        these turns are already gone). Prefer plan_eviction()/
        commit_eviction() directly whenever the evicted turns still need to
        be persisted into Tier 2 before being safely removed from Tier 1."""
        planned = self.plan_eviction(session_id, target_fraction)
        self.commit_eviction(session_id, len(planned))
        return planned
