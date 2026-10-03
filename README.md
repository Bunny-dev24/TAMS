# TAMS v2 — Tiered Agent Memory System with Lossless Fact Extraction

A 3-tier memory architecture for long-running agent conversations: raw
working memory, a raw-chunk long-session store (no narrative summarization),
and a structured, lossless fact store for 5 protected categories — fronted
by a hybrid (cosine + BM25 + LLM rerank) retrieval pipeline and a 3-agent
system (Orchestrator / Memory Manager / Verifier).

See [`tams/`](tams) for the implementation. This README covers setup,
architecture-to-code mapping, **why the retrieval pipeline is shaped the
way it is** (the part most memory systems get wrong or skip explaining),
running the eval harness, and known limitations — including the ones that
are still open, stated plainly rather than hand-waved.

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

## Why the retrieval funnel is shaped the way it is

This is the part of the system where almost all of the accuracy comes from,
so it's worth justifying each stage on its own terms instead of just
listing them. The funnel, concretely
([`tams/retrieval/hybrid.py`](tams/retrieval/hybrid.py)):

```
Tier 1 (full, raw, synchronous)         — always included, zero filtering
        +
Tier 3 entity-match hard facts          — deterministic, always included
        +
┌─────────────────────────────────────────────────────────────┐
│ cosine search (Tier2+3, up to pool_n)                        │
│         +   fused, decay-weighted            →  pool (≤2000) │
│ BM25 keyword search (same corpus)                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
ONE listwise LLM rerank call over the pool  →  final top-k
```

**Why cosine *and* BM25, fused, instead of just one?** They fail in opposite,
non-overlapping ways. Cosine similarity (embeddings) is good at paraphrase —
"the DB we picked" matches "we chose PostgreSQL" — but it's notoriously bad
at exact tokens: numbers, IDs, parameter names, and anything compressed away
by the embedding model (a timeout of `30` vs `45` can end up with a *higher*
cosine score to each other than either has to the literal query, because the
embedding space doesn't preserve numeric distance as semantic distance).
BM25 is the inverse: it's exact-token-perfect for "find the chunk containing
`max_retries`" but has zero paraphrase ability. TAMS fuses both
(`score = cos_score + 0.1 * bm25_score`) so a candidate only needs to win on
*either* axis to make the pool — this is why hybrid retrieval is now
standard practice in production RAG systems, not a TAMS-specific flourish.

**Why is the LLM call reserved for a small, already-filtered pool, and not
run per-candidate or over the raw corpus?** Two hard constraints collide
here: (1) LLM calls are the most expensive and slowest primitive in the
whole system per unit of work, so they must be *amortized* — one call doing
listwise comparison over N candidates, not N calls doing pointwise
classification; (2) an LLM asked to find a needle in an *undifferentiated*
haystack of thousands of unrelated turns does measurably worse than one
asked to rank an already-plausible shortlist — the cheap, parallelizable
cosine+BM25 stage exists specifically to produce that shortlist so the one
expensive call is spent on discrimination, not search. This is also why a
cross-encoder (e.g. bge-reranker) was considered and *rejected* for this
slot during design: cross-encoders score pointwise and context-free, with no
instruction-following and a training distribution (web passages) that
doesn't match terse structured fact text — the LLM reranker's listwise,
instruction-following, query-aware scoring is the actual reason reranking
helps here, and a cross-encoder would trade that away for latency it
doesn't currently need at this project's scale.

**Why does the LLM reranker get the final say over cosine/BM25 scores, but
Tier-1 and entity-matched facts bypass it entirely?** Any multi-stage
filter can only ever *maintain or reduce* recall relative to the stage
before it, never increase it — if the true answer isn't in the pool, no
reranker rescues it. Two things are deliberately exempted from that risk:
Tier 1 (the raw, current conversation) is never filtered at all, so the
model can never "forget" something from the last N turns regardless of how
retrieval performs; and Tier-3 facts whose entities literally match the
query's are included via deterministic lookup (`entity_match.py`),
bypassing similarity/rerank scoring entirely, so a hard numeric/decision
fact can never be mis-ranked out of context by an imperfect fusion score.

**Where this funnel is honestly *not* bulletproof:** the cosine+BM25 pool
is still a hard cutoff at `retrieval_pool_size` (default 2000) — if the
corpus for a session genuinely exceeds that and the true answer scores
outside the top 2000 on *both* cosine and BM25, it never reaches the
reranker. At the "single long-running session" scale this project targets,
that's not been an observed problem; it would need re-validation before
assuming it holds at a much larger multi-tenant scale.

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
  `chat()`/`chat_json()` degrade to a safe fallback string/`{}` after
  retries are exhausted rather than raising into the hot path.
- **Lossless eviction.** Tier 1 → Tier 2 compaction uses a non-destructive
  peek-then-commit pattern (`plan_eviction()` / `commit_eviction()` in
  [`tier1_working_memory.py`](tams/tiers/tier1_working_memory.py)): raw turns
  are only popped from Tier 1 *after* Tier 2 ingestion (chunking + anchored
  summary extension, which calls `embed()`) has durably succeeded. If that
  write fails for any reason (e.g. a transient OpenAI API outage) — Tier 1
  is left untouched and compaction simply retries on a later turn, instead
  of silently losing turns. This was found and fixed via real API testing,
  not theoretically.
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

## LLM provider

TAMS uses OpenAI directly (no other provider/gateway is supported). Set
`OPENAI_API_KEY` in `.env`; without it, TAMS runs fully in deterministic
fake-LLM mode (see below). A gateway/base-URL override (to route through a
third-party multi-model proxy such as Abacus AI RouteLLM) was evaluated and
deliberately removed — real OpenAI is the only backend going forward, kept
simple on purpose: one provider, one auth path, no gateway-specific quirks
to carry (e.g. some gateways were found, in testing, to omit embeddings
support entirely, or to return a `usage` object with `None` token-count
fields instead of omitting it — a defensive fix for the latter still lives
in `_usage_tokens()` in [`tams/llm/client.py`](tams/llm/client.py) since
it's correct regardless of provider).

A repeatable, non-mocked smoke test for the configured key lives at
[`scripts/smoke_test_provider.py`](scripts/smoke_test_provider.py) — it
exercises `chat()`, `chat_json()`, and `embed()` independently so a failure
in one doesn't hide results from the others:

```bash
python3 scripts/smoke_test_provider.py
```

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

- No benchmark against other memory systems (MemGPT, Mem0, Zep, LangMem,
  etc.) exists in this repo. Any claim about this system being "better"
  than those would be unsubstantiated marketing, not engineering — the
  design-rationale section above explains *why* each piece exists, which is
  a different and more honest claim than *better than X*.
- A FastAPI service layer is deliberately deferred until real-provider
  testing (above) is complete — today this is a CLI + library, not a
  served app.
- No RBAC, consent management, PII redaction, or automated data-retention
  policy exists yet in [`tams/governance/`](tams/governance) — only
  export/delete/correct with an audit trail (`CorrectionLog`). These are
  real gaps for anything beyond a single-user/dev context, not implemented
  gaps hidden behind vague wording.
- The MemGPT-style pure-summarization baseline
  (`tams/eval/baselines.py::memgpt_style_compress`) and the plain
  cosine-only RAG baseline are implemented as callable functions but aren't
  yet wired into a one-command side-by-side report with TAMS in the CLI
  (`run_full_context_baseline` is; the other two are usable directly from
  Python / easy follow-up CLI flags).
- `tams costs` reports from the ledger but doesn't yet have a persistent
  dashboard/UI — it's a `rich`-printed JSON report today.
- Retrieval pool size (`retrieval_pool_size`, default 2000) is a hard
  cutoff, not an adaptive one — see the retrieval-funnel section above for
  exactly where that could matter.
