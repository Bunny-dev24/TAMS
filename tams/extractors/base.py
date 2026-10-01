"""Base class for the 5 Tier-3 extractors. Each is a cheap, per-turn,
structured-JSON extraction call (mini-tier model) wrapped in a circuit
breaker — repeated failures auto-disable extraction for a cooldown window
rather than blocking the user-facing response (guardrail #7)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tams.config import Settings
from tams.llm.circuit_breaker import CircuitOpenError, get_breaker
from tams.llm.client import LLMClient
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore


class BaseExtractor(ABC):
    category: str

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings
        self.breaker = get_breaker(
            f"extractor:{self.category}",
            failure_threshold=settings.extractor_circuit_breaker_threshold,
            cooldown_s=settings.extractor_circuit_breaker_cooldown_s,
        )

    @abstractmethod
    def _schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def _prompt(self, turn: Turn, context_turns: list[Turn]) -> list[dict[str, str]]: ...

    @abstractmethod
    def _fake_fn(self, turn: Turn): ...

    @abstractmethod
    def _persist(self, fact_store: Tier3FactStore, turn: Turn, extracted: list[dict[str, Any]]) -> None: ...

    def run(self, turn: Turn, context_turns: list[Turn], fact_store: Tier3FactStore) -> list[dict[str, Any]]:
        """Returns [] (gracefully) if the circuit is open or extraction fails
        — extraction NEVER blocks or crashes the user-facing turn."""
        try:
            result = self.breaker.call(self._call_llm, turn, context_turns)
        except CircuitOpenError:
            return []
        except Exception:
            return []
        items = result if isinstance(result, list) else result.get("items", [])
        if items:
            try:
                self._persist(fact_store, turn, items)
            except Exception:
                # Persistence failure must not crash the turn either.
                return []
        return items

    def _call_llm(self, turn: Turn, context_turns: list[Turn]):
        messages = self._prompt(turn, context_turns)
        return self.llm_client.chat_json(
            messages,
            model=self.settings.extractor_model,
            schema=self._schema(),
            purpose=f"extractor:{self.category}",
            fake_fn=lambda _msgs: self._fake_fn(turn),
        )
