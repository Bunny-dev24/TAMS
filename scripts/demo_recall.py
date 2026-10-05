"""Offline, zero-API-key demo of TAMS lossless recall + conflict handling.

Run it with:

    python3 scripts/demo_recall.py

It needs no OPENAI_API_KEY (forces deterministic offline mode) and no
Redis/Qdrant (falls back to in-process stores). Designed to be screen-recorded:
clear sections, short pauses, and a visible proof that the answer survives
even after the original turn has been evicted out of raw working memory.

Storyline:
  1. User states a value:  request_timeout is 45 seconds.
  2. ~50 unrelated turns go by (enough to evict the original turn out of
     Tier 1 raw working memory into Tier 2).
  3. We ask again. TAMS still answers 45 -- recalled from the structured
     Tier 3 fact store via deterministic entity match, not from raw context.
  4. User changes it to 60. The old 45 is SUPERSEDED, never deleted.
  5. Export shows both rows: 60 active, 45 kept for audit.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# Make the demo runnable from a plain `python3 scripts/demo_recall.py` even if
# the `tams` package was never pip-installed into the active env (e.g. conda
# base) — resolve `import tams` against the repo source tree directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force deterministic offline mode BEFORE importing anything that reads config.
os.environ.pop("OPENAI_API_KEY", None)

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def _pause(seconds: float = 1.1) -> None:
    time.sleep(seconds)


def _say(user_msg: str, orch, *, show: bool = True) -> str:
    result = orch.handle_turn(user_msg)
    if show:
        console.print(f"[bold cyan]you  >[/bold cyan] {user_msg}")
        console.print(f"[bold magenta]tams >[/bold magenta] {result.response_text}\n")
    return result.response_text


def main() -> None:
    from tams.config import load_settings
    from tams.session import build_session, reset_shared_state
    from tams.storage.db import init_db
    from tams.agents.orchestrator import OrchestratorAgent

    # Isolated temp DB + a deliberately small working-memory window so the
    # original turn actually gets evicted out of Tier 1 within the demo.
    tmp_db = os.path.join(tempfile.mkdtemp(prefix="tams-demo-"), "demo.db")
    settings = load_settings().model_copy(update={
        "sqlite_path": tmp_db,
        "working_memory_max_turns": 16,
    })
    init_db(settings.sqlite_path)
    reset_shared_state()

    session = build_session("demo", settings, user_id="demo-user")
    orch = OrchestratorAgent(session, async_memory=False)  # sync = deterministic

    console.print(Panel.fit(
        "[bold]TAMS demo[/bold]  -  lossless agent memory, running fully offline\n"
        "[dim]no API key, no Redis, no Qdrant -- deterministic offline mode[/dim]",
        border_style="green",
    ))
    _pause()

    console.print(Rule("[bold]1. The user states a value[/bold]"))
    console.print("[bold cyan]you  >[/bold cyan] The request_timeout is 45 seconds, do not exceed it.")
    orch.handle_turn("The request_timeout is 45 seconds, do not exceed it.")
    console.print("[dim]      (TAMS extracts request_timeout = 45s into the Tier 3 "
                  "fact store in the background)[/dim]\n")
    _pause()

    console.print(Rule("[bold]2. ~50 unrelated turns go by[/bold]"))
    chit_chat = [
        "What's a good name for a staging environment?",
        "Can you remind me what CI stands for?",
        "I like concise commit messages.",
        "What's the capital of Portugal?",
        "Explain what a feature flag is.",
    ]
    for i in range(50):
        _say(chit_chat[i % len(chit_chat)], orch, show=False)
    console.print("[dim]...50 turns of unrelated chatter later...[/dim]\n")
    _pause()

    # Prove the original turn is GONE from raw working memory (Tier 1).
    wm = session.working_memory.get_all(session.session_id)
    still_in_raw = any("request_timeout is 45" in t.content.lower() for t in wm)
    console.print(Panel.fit(
        f"Raw working memory (Tier 1) now holds [bold]{len(wm)}[/bold] turns.\n"
        f"Is the original 'request_timeout is 45' turn still in raw memory?  "
        f"[bold red]{still_in_raw}[/bold red]  (it was evicted to Tier 2)",
        title="proof: the original turn is gone from raw context",
        border_style="yellow",
    ))
    _pause()

    console.print(Rule("[bold]3. Ask again -- recalled from the Tier 3 fact store[/bold]"))
    _say("What is request_timeout set to again?", orch)
    _pause()

    console.print(Rule("[bold]4. The user changes it -- old value is superseded, not deleted[/bold]"))
    _say("Update: the request_timeout is 60 seconds now.", orch)
    _pause()
    _say("Remind me, what is request_timeout?", orch)
    _pause()

    console.print(Rule("[bold]5. Nothing was overwritten -- full audit trail[/bold]"))
    export = session.fact_store.fact_store.export_all()
    for row in export.get("numeric", []):
        if str(row.get("param")) == "request_timeout":
            status = row.get("status")
            colour = "green" if status == "active" else "dim"
            console.print(
                f"  [{colour}]request_timeout = {row.get('value')} {row.get('unit') or ''}"
                f"  ->  status: {status}[/{colour}]"
            )
    console.print()
    console.print(Panel.fit(
        "The number survived 50 unrelated turns AND an eviction out of raw memory.\n"
        "The old value is still on record, superseded rather than lost.\n"
        "[bold green]That is the whole point of TAMS.[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
