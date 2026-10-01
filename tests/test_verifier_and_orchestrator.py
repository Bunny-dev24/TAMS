from tams.agents.orchestrator import OrchestratorAgent
from tams.agents.verifier import VerifierAgent
from tams.llm.client import LLMClient
from tams.session import build_session


def test_verifier_flags_unsupported_number(settings):
    llm = LLMClient(settings)
    verifier = VerifierAgent(llm, settings)
    context = "The timeout is 300 ms and the budget is 5000 usd."
    response = "The timeout is 300 ms, and the budget is 999999 usd."
    report = verifier.verify(response, context)
    assert any("999999" in v for v in report.rule_violations)


def test_verifier_citation_tagging(settings):
    llm = LLMClient(settings)
    verifier = VerifierAgent(llm, settings)
    context = "The project uses [mem:abc123] PostgreSQL for storage."
    response = "The project uses PostgreSQL for storage."
    report = verifier.verify(response, context)
    assert report.citation_coverage >= 0.0
    assert isinstance(report.tagged_text, str)


def test_orchestrator_end_to_end_turn(settings):
    session = build_session("orch-test", settings)
    orch = OrchestratorAgent(session, async_memory=False)

    r1 = orch.handle_turn("The monthly_budget is set to 5000 usd and we must never exceed it.")
    assert r1.turn_id == 1
    assert not r1.retrieval.degraded

    r2 = orch.handle_turn("What is the monthly_budget?")
    assert any(f.get("param") == "monthly_budget" for f in r2.retrieval.hard_facts)
    assert "5000" in r2.response_text


def test_orchestrator_conflict_resolution_prefers_recent(settings):
    session = build_session("orch-conflict", settings)
    orch = OrchestratorAgent(session, async_memory=False)

    orch.handle_turn("The max_retries is set to 3.")
    orch.handle_turn("Actually, the max_retries is set to 10.")
    r3 = orch.handle_turn("What is max_retries?")

    numeric_hits = [f for f in r3.retrieval.hard_facts if f.get("param") == "max_retries"]
    assert len(numeric_hits) == 1
    assert numeric_hits[0]["value"] == "10"
