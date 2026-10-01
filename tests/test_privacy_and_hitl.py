from tams.agents.orchestrator import OrchestratorAgent
from tams.governance.hitl import correct_fact, correction_history
from tams.governance.privacy import delete_session, export_session
from tams.session import build_session


def test_export_session_includes_all_tiers(settings):
    session = build_session("priv-1", settings)
    orch = OrchestratorAgent(session, async_memory=False)
    orch.handle_turn("The max_retries is set to 7.")

    data = export_session(session)
    assert data["session_id"] == "priv-1"
    assert len(data["tier1_working_memory"]) >= 1
    assert any(f["param"] == "max_retries" for f in data["tier3_facts"]["numeric"])


def test_delete_session_removes_everything(settings):
    session = build_session("priv-2", settings)
    orch = OrchestratorAgent(session, async_memory=False)
    orch.handle_turn("The max_retries is set to 7.")

    result = delete_session(session)
    assert result["facts_deleted"] >= 1

    data = export_session(session)
    assert data["tier1_working_memory"] == []
    assert all(len(v) == 0 for v in data["tier3_facts"].values())


def test_hitl_correction_updates_retrieval(settings):
    session = build_session("hitl-1", settings)
    orch = OrchestratorAgent(session, async_memory=False)
    r1 = orch.handle_turn("The monthly_budget is set to 100 usd.")

    numeric_facts = session.fact_store.fact_store.active_facts("numeric")
    budget_fact = next(f for f in numeric_facts if f["param"] == "monthly_budget")

    new_id = correct_fact(session, "numeric", budget_fact["id"], {"value": "500"}, note="user said wrong amount")

    active = session.fact_store.fact_store.active_facts("numeric")
    corrected = next(f for f in active if f["param"] == "monthly_budget")
    assert corrected["value"] == "500"
    assert corrected["id"] == new_id

    history = correction_history(budget_fact["id"])
    assert len(history) == 1
    assert history[0]["new_value"]["value"] == "500"
