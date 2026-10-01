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

    def evict_for_compaction(self, session_id: str, target_fraction: float = 0.5) -> list[Turn]:
        """Evict oldest turns until usage drops to target_fraction of budget
        (or the turn-count cap). Returns evicted turns (handed to Tier 2)."""
        target_tokens = int(self.budget_tokens() * target_fraction)
        evicted: list[Turn] = []
        turns = self.store.get_all(session_id)
        while turns and (
            self.current_tokens(session_id) > target_tokens
            or len(self.store.get_all(session_id)) > self.settings.working_memory_max_turns
        ):
            batch = self.store.pop_oldest(session_id, self.settings.chunk_size_turns)
            if not batch:
                break
            evicted.extend(batch)
            turns = self.store.get_all(session_id)
        return evicted
