import time

import pytest

from tams.llm.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


def test_breaker_opens_after_threshold_failures():
    cb = CircuitBreaker(name="t", failure_threshold=3, cooldown_s=60)

    def fail():
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(fail)

    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(fail)


def test_breaker_half_opens_after_cooldown_and_closes_on_success():
    cb = CircuitBreaker(name="t2", failure_threshold=1, cooldown_s=0.05)

    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cb.call(fail)
    assert cb.state == CircuitState.OPEN

    time.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN

    result = cb.call(lambda: "ok")
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


def test_extraction_never_crashes_on_open_circuit(settings):
    from tams.extractors.numeric import NumericExtractor
    from tams.llm.client import LLMClient
    from tams.storage.working_memory import Turn
    from tams.tiers.tier3_fact_store import Tier3FactStore
    from tams.storage.vector_store import InMemoryVectorStore
    from tams.storage.bm25_index import BM25Index

    llm = LLMClient(settings)
    extractor = NumericExtractor(llm, settings)
    extractor.breaker._state = CircuitState.OPEN
    extractor.breaker._opened_at = time.monotonic()
    extractor.breaker.cooldown_s = 999

    fact_store = Tier3FactStore("s1", InMemoryVectorStore(), BM25Index(), llm)
    turn = Turn(turn_id=1, session_id="s1", speaker="user", content="budget is 100 usd")
    result = extractor.run(turn, [], fact_store)
    assert result == []  # degrades gracefully, never raises
