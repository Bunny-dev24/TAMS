# TAMS v2 — Tiered Agent Memory System with Lossless Fact Extraction

A 3-tier memory architecture for long-running agent conversations: raw
working memory, a raw-chunk long-session store (no narrative summarization),
and a structured, lossless fact store for 5 protected categories — fronted
by a hybrid (cosine + BM25 + LLM rerank) retrieval pipeline and a 3-agent
system (Orchestrator / Memory Manager / Verifier).

See [`tams/`](tams) for the implementation. This README covers setup,
architecture-to-code mapping, running the eval harness, and known
limitations.

## Quickstart

```bash
# 1. Install (editable) — a venv is recommended; this sandbox used:
pip install --user --break-system-packages -e ".[dev]"

# 2. (Optional) copy env and set your OpenAI key — without a key, TAMS runs
#    in a fully deterministic, zero-network "fake LLM" mode (dev/CI/offline).
cp .env.example .env

# 3. (Optional) bring up Redis + Qdrant for production-grade Tier 1/2 storage.
#    Without them, TAMS gracefully falls back to in-process stores.
docker compose up -d

# 4. Run the test suite
pytest -q

# 5. Chat
python -m tams.cli chat

# 6. Run the needle-in-haystack eval
python -m tams.cli eval --num-filler-turns 100 --ablation none
```

## Architecture → code map

| Spec section | Code |
|---|---|
| Tier 1 — Working Memory | [`tams/tiers/tier1_working_memory.py`](tams/tiers/tier1_working_memory.py), [`tams/storage/working_memory.py`](tams/storage/working_memory.py) (Redis / in-memory) |
| Tier 2 — Raw Chunk Store + anchored summary | [`tams/tiers/tier2_chunk_store.py`](tams/tiers/tier2_chunk_store.py) |
| Tier 3 — Structured Fact Store | [`tams/storage/fact_db.py`](tams/storage/fact_db.py), [`tams/tiers/tier3_fact_store.py`](tams/tiers/tier3_fact_store.py), [`tams/storage/models.py`](tams/storage/models.py) |
| 5 extractors | [`tams/extractors/`](tams/extractors) (numeric, constraint, decision, dependency, preference) |
| Hybrid retrieval + rerank | [`tams/retrieval/hybrid.py`](tams/retrieval/hybrid.py), [`reranker.py`](tams/retrieval/reranker.py), [`entity_match.py`](tams/retrieval/entity_match.py) |
| Orchestrator / Memory Manager / Verifier | [`tams/agents/`](tams/agents) |
| Conflict resolution + decay | `FactStore.write()` in [`fact_db.py`](tams/storage/fact_db.py), [`tams/decay.py`](tams/decay.py) |
| Circuit breakers | [`tams/llm/circuit_breaker.py`](tams/llm/circuit_breaker.py) |
| Cost monitoring | [`tams/storage/cost_ledger.py`](tams/storage/cost_ledger.py), `tams costs` CLI |
| Multi-session identity | [`tams/identity/profile_store.py`](tams/identity/profile_store.py) |
| Privacy (export/delete) | [`tams/governance/privacy.py`](tams/governance/privacy.py), `tams export` / `tams delete` |
| HITL correction | [`tams/governance/hitl.py`](tams/governance/hitl.py), `tams correct` |
| Eval harness | [`tams/eval/`](tams/eval) (synthetic needle-in-haystack, metrics, baselines) |

## Key design choices / trade-offs

- **Storage backends.** Tier 1 = Redis (fallback: in-process dict).
  Tier 2/3 vectors = Qdrant (fallback: in-process numpy cosine index). Tier 3
  structured facts = always-embedded SQLite (no separate server needed —
  it's relational data that benefits from real queries, not a vector
  problem). BM25 = in-process `rank_bm25` over the same corpus that gets
  embedded, rather than standing up a separate search engine (Elasticsearch)
  — simpler ops for the same hybrid-retrieval benefit.
- **Graceful degradation** is implemented at every storage boundary
  (`build_vector_store`, `build_working_memory_store`) and at the top of the
  retrieval pipeline (`HybridRetriever.retrieve` catches everything and
  falls back to Tier-1-only). Extractors never raise — the circuit breaker
  (`CircuitOpenError`) is caught internally and extraction returns `[]`.
- **Verifier is a *bounded* hot-path step, not a background one.** The spec
  asks for a post-generation groundedness check (which the user is waiting
  on) but also for "no recursive LLM calls in the hot path, only async
  background." We resolve this by making the rules layer pure Python
  (free), and the judge layer exactly one LLM call over all sentences at
  once, with **at most one** regeneration pass if the hallucination rate
  exceeds `hallucination_regen_threshold`. It's finite/bounded, never an
  open-ended loop — but it does happen before the response returns. The 5
  extractors + anchored-summary maintenance, by contrast, are genuinely
  fire-and-forget on a background `ThreadPoolExecutor`
  (`MemoryManager.dispatch_async`).
- **Eventual consistency.** Because extraction is async, a fact stated in
  turn N may not be queryable via the deterministic entity-lookup until
  slightly after turn N's response is returned (it's being written in the
  background). This is intentional — it's the cost of never blocking the
  user-facing path on extraction. Tier 1 (full, raw, synchronous) is what
  guarantees the orchestrator always sees the current turn regardless.
- **Conflict resolution** happens inside `FactStore.write()`: writing a fact
  whose key field (e.g. `param` for numeric, `entity` for dependency)
  already has an *active* row with a *different* value marks the old row
  `superseded` (never deleted) and the new one `active`. Re-stating the
  *same* value is treated as idempotent (bumps usage/recency, no duplicate
  row). HITL corrections (`FactStore.correct`) follow the same
  supersede-never-delete pattern plus a `CorrectionLog` audit entry.
- **Decay** (`tams/decay.py`) is an exponential recency+usage multiplier
  applied only at *retrieval-ranking* time — it never deletes or hides rows
  from direct queries (`active_facts`, exports), only from the
  decay-weighted ranking used during hybrid retrieval scoring.

## Fake-LLM mode (no API key) — what to expect

Every LLM call (`chat`, `chat_json`, `embed`) routes through
[`tams/llm/client.py`](tams/llm/client.py), which uses a fully deterministic
`FakeLLMBackend` when `OPENAI_API_KEY` is unset. This makes the entire
system (orchestrator, all 5 extractors, reranker, verifier, eval harness)
runnable offline with zero network calls and zero cost — useful for CI and
for exploring the architecture without an API key.

**Important caveat:** fake embeddings are hash-seeded deterministic vectors
with no real semantic content. BM25 (lexical) and the deterministic
entity-match lookup still work correctly in fake mode, but **cosine
similarity carries no actual meaning** — a query like *"Which database was
chosen and why?"* won't semantically match a stored fact *"we chose
PostgreSQL over MongoDB"* unless they share literal keywords, because there's
no real embedding model behind the "similarity" score. With a real
`OPENAI_API_KEY`, this is exactly the gap genuine embeddings close, and it's
part of why the eval harness is useful: run it once in fake mode (free,
structural sanity check) and once with a real key (true semantic recall
numbers) to see the delta.

## Eval harness

```bash
python -m tams.cli eval --num-filler-turns 100 --ablation none
python -m tams.cli eval --num-filler-turns 100 --ablation no-rerank
python -m tams.cli eval --num-filler-turns 100 --ablation no-extractors
python -m tams.cli eval --num-filler-turns 100 --ablation no-decay
```

Reports recall@k **per protected category** (numeric / constraint /
decision / dependency), plus hallucination rate, citation coverage,
retrieval latency, and the Tier1→Tier2 compaction token-reduction ratio,
against a full-context-stuffing baseline. `working_memory_max_turns` is
deliberately small by default in the CLI eval command so that compaction
actually fires within the synthetic haystack — otherwise, with the
production-sized context window, nothing gets evicted and every ablation
trivially scores 1.0 (Tier 1 "full" already contains everything).

## What's intentionally out of scope for this first pass

- The MemGPT-style pure-summarization baseline
  (`tams/eval/baselines.py::memgpt_style_compress`) and the plain
  cosine-only RAG baseline are implemented as callable functions but aren't
  yet wired into a one-command side-by-side report with TAMS in the CLI
  (`run_full_context_baseline` is; the other two are usable directly from
  Python / easy follow-up CLI flags).
- `tams costs` reports from the ledger but doesn't yet have a persistent
  dashboard/UI — it's a `rich`-printed JSON report today.
