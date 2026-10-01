from tams.extractors.constraint import ConstraintExtractor
from tams.extractors.decision import DecisionExtractor
from tams.extractors.dependency import DependencyExtractor
from tams.extractors.numeric import NumericExtractor
from tams.extractors.preference import PreferenceDetector
from tams.llm.client import LLMClient
from tams.storage.bm25_index import BM25Index
from tams.storage.vector_store import InMemoryVectorStore
from tams.storage.working_memory import Turn
from tams.tiers.tier3_fact_store import Tier3FactStore


def _fact_store(settings):
    llm = LLMClient(settings)
    return llm, Tier3FactStore("s1", InMemoryVectorStore(), BM25Index(), llm)


def test_numeric_extractor_persists_fact(settings):
    llm, fs = _fact_store(settings)
    ext = NumericExtractor(llm, settings)
    turn = Turn(turn_id=1, session_id="s1", speaker="user", content="the request_timeout is set to 300 ms")
    items = ext.run(turn, [], fs)
    assert len(items) == 1
    facts = fs.fact_store.active_facts("numeric")
    assert facts[0]["param"] == "request_timeout"
    assert facts[0]["value"] == "300"
    assert facts[0]["unit"] == "ms"


def test_constraint_extractor_captures_must_not(settings):
    llm, fs = _fact_store(settings)
    ext = ConstraintExtractor(llm, settings)
    turn = Turn(turn_id=1, session_id="s1", speaker="user", content="The export job must never run during business hours because it locks the DB.")
    items = ext.run(turn, [], fs)
    assert len(items) == 1
    facts = fs.fact_store.active_facts("constraint")
    assert "must never run during business hours" in facts[0]["constraint_text"]
    assert "locks the db" in facts[0]["reason"].lower()


def test_decision_extractor_captures_rejected_alternative_and_reason(settings):
    llm, fs = _fact_store(settings)
    ext = DecisionExtractor(llm, settings)
    turn = Turn(
        turn_id=1,
        session_id="s1",
        speaker="assistant",
        content="We decided to go with PostgreSQL instead of MongoDB because we needed strong transactions.",
    )
    items = ext.run(turn, [], fs)
    assert len(items) == 1
    facts = fs.fact_store.active_facts("decision")
    assert "PostgreSQL" in facts[0]["decision"]
    assert any("MongoDB" in alt for alt in facts[0]["rejected_alternatives"])
    assert "transactions" in facts[0]["reason"]


def test_dependency_extractor_modification_then_reference(settings):
    llm, fs = _fact_store(settings)
    ext = DependencyExtractor(llm, settings)
    t1 = Turn(turn_id=1, session_id="s1", speaker="user", content="I renamed the AuthModule to IdentityModule.")
    ext.run(t1, [], fs)

    facts = fs.fact_store.active_facts("dependency")
    assert len(facts) == 1
    assert facts[0]["entity"] == "AuthModule"
    assert facts[0]["modified_at_turn"] == 1

    t2 = Turn(turn_id=5, session_id="s1", speaker="user", content="Does AuthModule still handle sessions?")
    ext.run(t2, [], fs)
    facts = fs.fact_store.active_facts("dependency")
    assert 5 in facts[0]["referenced_at_turns"]


def test_preference_detector_batch_runs_only_when_interval_reached(settings):
    llm, fs = _fact_store(settings)
    detector = PreferenceDetector(llm, settings)
    assert detector.should_run(settings.preference_batch_interval_turns - 1) is False
    assert detector.should_run(settings.preference_batch_interval_turns) is True

    turns = [
        Turn(turn_id=i, session_id="s1", speaker="user", content="pytest pytest pytest please")
        for i in range(1, 5)
    ]
    items = detector.run_batch(turns, fs)
    assert any(i["pref_type"] == "preferred_tool" for i in items)
    facts = fs.fact_store.active_facts("preference")
    assert any(f["pref_type"] == "preferred_tool" and f["value"] == "pytest" for f in facts)
