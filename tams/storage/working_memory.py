"""Tier 1 storage backend: raw turns, last-N, unmodified. Redis in prod,
in-process fallback if Redis is unreachable (graceful degradation)."""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Turn:
    turn_id: int
    session_id: str
    speaker: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "speaker": self.speaker,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Turn":
        return Turn(**d)


class WorkingMemoryStore(ABC):
    @abstractmethod
    def append(self, turn: Turn) -> None: ...

    @abstractmethod
    def get_all(self, session_id: str) -> list[Turn]: ...

    @abstractmethod
    def pop_oldest(self, session_id: str, n: int) -> list[Turn]:
        """Evict and return the n oldest turns (for Tier-2 handoff)."""

    @abstractmethod
    def healthy(self) -> bool: ...


class InMemoryWorkingMemoryStore(WorkingMemoryStore):
    def __init__(self):
        self._data: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def append(self, turn: Turn) -> None:
        with self._lock:
            self._data.setdefault(turn.session_id, []).append(turn)

    def get_all(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._data.get(session_id, []))

    def pop_oldest(self, session_id: str, n: int) -> list[Turn]:
        with self._lock:
            turns = self._data.get(session_id, [])
            evicted, remainder = turns[:n], turns[n:]
            self._data[session_id] = remainder
            return evicted

    def healthy(self) -> bool:
        return True


class RedisWorkingMemoryStore(WorkingMemoryStore):
    def __init__(self, url: str):
        import redis

        self._r = redis.from_url(url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"tams:tier1:{session_id}"

    def append(self, turn: Turn) -> None:
        self._r.rpush(self._key(turn.session_id), json.dumps(turn.to_dict()))

    def get_all(self, session_id: str) -> list[Turn]:
        raw = self._r.lrange(self._key(session_id), 0, -1)
        return [Turn.from_dict(json.loads(x)) for x in raw]

    def pop_oldest(self, session_id: str, n: int) -> list[Turn]:
        key = self._key(session_id)
        evicted = []
        for _ in range(n):
            raw = self._r.lpop(key)
            if raw is None:
                break
            evicted.append(Turn.from_dict(json.loads(raw)))
        return evicted

    def healthy(self) -> bool:
        try:
            self._r.ping()
            return True
        except Exception:
            return False


def build_working_memory_store(url: str) -> WorkingMemoryStore:
    try:
        store = RedisWorkingMemoryStore(url)
        if store.healthy():
            return store
    except Exception:
        pass
    return InMemoryWorkingMemoryStore()
