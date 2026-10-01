from tams.llm.client import LLMClient
from tams.storage.cost_ledger import make_cost_sink, report


def test_cost_sink_logs_calls_and_report_aggregates(settings):
    settings.cost_input_per_1k = 1.0
    settings.cost_output_per_1k = 2.0
    sink = make_cost_sink("session-x", settings)
    llm = LLMClient(settings, cost_sink=sink)

    llm.chat([{"role": "user", "content": "hello world"}], model="m", purpose="orchestrator")
    llm.embed(["some text to embed"], purpose="tier2_chunk_embed")

    r = report(session_id="session-x")
    assert r["total_calls"] == 2
    assert "orchestrator" in r["by_purpose"]
    assert "tier2_chunk_embed" in r["by_purpose"]
    assert r["total_cost_usd"] >= 0
    assert r["cost_per_interaction_usd"] == r["total_cost_usd"] / 2
