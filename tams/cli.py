"""TAMS CLI: interactive chat REPL, eval harness runner, cost report,
privacy export/delete, and HITL fact correction."""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from tams.config import settings as default_settings
from tams.storage.db import init_db

app = typer.Typer(help="TAMS v2 — Tiered Agent Memory System")
console = Console()


@app.command()
def chat(session_id: str = "cli-session", user_id: str = "default"):
    """Interactive REPL against the Orchestrator Agent."""
    init_db(default_settings.sqlite_path)
    from tams.agents.orchestrator import OrchestratorAgent
    from tams.session import build_session

    session = build_session(session_id, default_settings, user_id=user_id)
    orchestrator = OrchestratorAgent(session)

    if session.llm_client.is_fake:
        console.print("[yellow]No OPENAI_API_KEY set — running in deterministic fake-LLM mode.[/yellow]")

    console.print(f"[bold green]TAMS chat[/bold green] (session={session_id}). Ctrl-D to exit.")
    while True:
        try:
            msg = console.input("[bold cyan]you>[/bold cyan] ")
        except EOFError:
            break
        if not msg.strip():
            continue
        result = orchestrator.handle_turn(msg)
        console.print(f"[bold magenta]tams>[/bold magenta] {result.tagged_text}")
        if result.verification.rule_violations:
            console.print(f"[red]rule violations: {result.verification.rule_violations}[/red]")
        if result.retrieval.degraded:
            console.print(f"[yellow]degraded (retrieval fallback): {result.retrieval.error}[/yellow]")


@app.command()
def eval(
    num_filler_turns: int = 200,
    seed: int = 42,
    ablation: str = typer.Option("none", help="none|no-rerank|no-extractors|no-decay"),
    working_memory_max_turns: int = typer.Option(
        30, help="Small on purpose: forces Tier1->Tier2 eviction within the haystack so ablations are meaningful."
    ),
):
    """Run the needle-in-haystack eval harness.

    Uses a small working_memory_max_turns by default (relative to
    num_filler_turns) so Tier 1 -> Tier 2 compaction actually happens during
    the run — otherwise, with the production-sized context window, nothing
    ever gets evicted and every ablation trivially scores 1.0 recall@k
    (Tier 1 'full' already contains everything).
    """
    init_db(default_settings.sqlite_path)
    from tams.eval.harness import run_full_context_baseline, run_tams_eval
    from tams.eval.synth_data import generate_haystack

    eval_settings = default_settings.model_copy(update={"working_memory_max_turns": working_memory_max_turns})
    haystack = generate_haystack(num_filler_turns=num_filler_turns, seed=seed)

    kwargs = {"enable_hybrid_rerank": True, "enable_extractors": True, "enable_decay": True}
    if ablation == "no-rerank":
        kwargs["enable_hybrid_rerank"] = False
    elif ablation == "no-extractors":
        kwargs["enable_extractors"] = False
    elif ablation == "no-decay":
        kwargs["enable_decay"] = False

    console.print(f"[bold]Running TAMS eval[/bold] (ablation={ablation}, needles={len(haystack.needles)})...")
    tams_report = run_tams_eval(haystack, eval_settings, **kwargs)
    baseline_report = run_full_context_baseline(haystack, eval_settings)

    table = Table(title="Recall@k by category")
    table.add_column("category")
    table.add_column("TAMS recall@k")
    table.add_column("full-context-stuffing recall@k")
    categories = set(tams_report.recall_by_category) | set(baseline_report.recall_by_category)
    for cat in sorted(categories):
        tams_r = tams_report.recall_by_category.get(cat)
        base_r = baseline_report.recall_by_category.get(cat)
        table.add_row(
            cat,
            f"{tams_r.recall_at_k:.2f}" if tams_r else "-",
            f"{base_r.recall_at_k:.2f}" if base_r else "-",
        )
    console.print(table)
    console.print(json.dumps(tams_report.to_dict(), indent=2))


@app.command()
def costs(session_id: str = typer.Option(None), since_hours: float = typer.Option(None)):
    """Print the cost-per-interaction report."""
    init_db(default_settings.sqlite_path)
    from tams.storage.cost_ledger import report

    r = report(session_id=session_id, since_hours=since_hours)
    console.print_json(data=r)


@app.command()
def export(user_id: str):
    """GDPR-style export of all stored data for a user."""
    init_db(default_settings.sqlite_path)
    from tams.governance.privacy import export_user

    data = export_user(user_id, default_settings)
    console.print_json(data=data)


@app.command()
def delete(user_id: str, yes: bool = typer.Option(False, "--yes", help="skip confirmation")):
    """Right-to-forget: hard-delete all stored data for a user."""
    init_db(default_settings.sqlite_path)
    if not yes:
        confirmed = typer.confirm(f"Permanently delete all TAMS data for user '{user_id}'?")
        if not confirmed:
            raise typer.Abort()
    from tams.governance.privacy import delete_user

    result = delete_user(user_id, default_settings)
    console.print_json(data=result)


@app.command()
def correct(session_id: str, category: str, fact_id: str, field: str, value: str, note: str = ""):
    """Human-in-the-loop correction of a single stored fact field."""
    init_db(default_settings.sqlite_path)
    from tams.governance.hitl import correct_fact
    from tams.session import build_session

    session = build_session(session_id, default_settings)
    new_id = correct_fact(session, category, fact_id, {field: value}, note=note or None)
    console.print(f"[green]Corrected. New fact id: {new_id}[/green]")


if __name__ == "__main__":
    app()
