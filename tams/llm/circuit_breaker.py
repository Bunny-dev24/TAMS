"""Simple circuit breaker for extractor / reranker / verifier LLM calls.

Guardrail #7: "if extraction API fails repeatedly, auto-disable and continue
without memory rather than blocking responses" (Mem0-style production pattern).

States: CLOSED (normal) -> OPEN (failing, calls short-circuited) -> HALF_OPEN
(cooldown elapsed, allow one trial call) -> CLOSED or back to OPEN.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    cooldown_s: float = 60.0

    _failures: int = field(default=0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and (
                time.monotonic() - self._opened_at >= self.cooldown_s
            ):
                self._state = CircuitState.HALF_OPEN
            return self._state

    def allow(self) -> bool:
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def call(self, fn, *args, **kwargs):
        """Run fn under breaker protection. Raises CircuitOpenError if open."""
        if not self.allow():
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(name: str, failure_threshold: int = 5, cooldown_s: float = 60.0) -> CircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name, failure_threshold=failure_threshold, cooldown_s=cooldown_s
            )
        return _breakers[name]


def reset_all_breakers() -> None:
    """Test helper: clears all cached breakers so each test starts fresh."""
    with _breakers_lock:
        _breakers.clear()
