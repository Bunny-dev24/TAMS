"""The hybrid retrieval pipeline — per the design philosophy, this is where
engineering effort should concentrate (the ~20pt accuracy swing comes from
hybrid+rerank, not write-side sophistication).

    query -> embed
    parallel: cosine (Tier2+3) + BM25 keyword search
    pool top-2k candidates from both -> LLM reranker picks final top-k
    ALWAYS include: Tier1 (full) + any Tier3 facts matching query entities
        (deterministic lookup, not just similarity)
    assemble: working memory + reranked chunks + hard facts

Graceful degradation: any failure anywhere in this pipeline falls back to
Tier-1-only context — retrieval never crashes/blocks the user-facing turn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from tams.config import Settings
from tams.decay import decay_weight
from tams.retrieval.embeddings import CachedEmbedder
from tams.retrieval.entity_match import extract_query_entities
from tams.retrieval.reranker import Candidate, LLMReranker
from tams.storage.bm25_index import BM25Index
from tams.storage.vector_store import VectorStore
from tams.tiers.tier1_working_memory import WorkingMemory
from tams.tiers.tier2_chunk_store import CHUNK_COLLECTION
from tams.tiers.tier3_fact_store import FACT_COLLECTION, Tier3FactStore
from tams.tokens import count_tokens


@dataclass
class RetrievalResult:
    working_memory_text: str
    reranked_items: list[Candidate]
    hard_facts: list[dict]
    context_text: str
    latency_s: float
    degraded: bool = False
    error: str | None = None


class HybridRetriever:
    def __init__(
        self,
        working_memory: WorkingMemory,
        vector_store: VectorStore,
        bm25: BM25Index,
        embedder: CachedEmbedder,
        reranker: LLMReranker,
        settings: Settings,
        enable_hybrid_rerank: bool = True,
    ):
        self.working_memory = working_memory
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self.enable_hybrid_rerank = enable_hybrid_rerank  # ablation switch

    def retrieve(
        self,
        session_id: str,
        query: str,
        current_turn: int,
        fact_store: Tier3FactStore,
        enable_decay: bool = True,
    ) -> RetrievalResult:
        t0 = time.monotonic()
        wm_turns = self.working_memory.get_all(session_id)
        wm_text = "\n".join(f"{t.speaker} (turn {t.turn_id}): {t.content}" for t in wm_turns)

        try:
            reranked, hard_facts = self._retrieve_tier23(
                session_id, query, current_turn, fact_store, enable_decay
            )
        except Exception as e:  # graceful degradation guardrail
            context_text = self._assemble(wm_text, [], [])
            return RetrievalResult(
                working_memory_text=wm_text,
                reranked_items=[],
                hard_facts=[],
                context_text=context_text,
                latency_s=time.monotonic() - t0,
                degraded=True,
                error=str(e),
            )

        context_text = self._assemble(wm_text, reranked, hard_facts)
        return RetrievalResult(
            working_memory_text=wm_text,
            reranked_items=reranked,
            hard_facts=hard_facts,
            context_text=context_text,
            latency_s=time.monotonic() - t0,
            degraded=False,
        )

    # ------------------------------------------------------------------
    def _retrieve_tier23(
        self,
        session_id: str,
        query: str,
        current_turn: int,
        fact_store: Tier3FactStore,
        enable_decay: bool,
    ) -> tuple[list[Candidate], list[dict]]:
        # 1. Deterministic entity lookup — ALWAYS included regardless of
        #    similarity ranking.
        entities = extract_query_entities(query)
        hard_facts = fact_store.match_entities(entities) if entities else []

        if not self.enable_hybrid_rerank:
            # Ablation: cosine-only, no BM25/rerank.
            qvec = self.embedder.embed_query(query, self.settings.embedding_model)
            hits = self.vector_store.search(CHUNK_COLLECTION, qvec, self.settings.retrieval_top_k)
            cands = [Candidate(id=h.id, text=h.payload.get("text", ""), source="tier2", prelim_score=h.score) for h in hits]
            return cands[: self.settings.retrieval_top_k], hard_facts

        # 2. Embed query (cached).
        qvec = self.embedder.embed_query(query, self.settings.embedding_model)

        # 3. Parallel cosine search over Tier2 chunks + Tier3 facts.
        pool_n = self.settings.retrieval_pool_size
        chunk_hits = self.vector_store.search(CHUNK_COLLECTION, qvec, pool_n, filter={"session_id": session_id})
        fact_hits = self.vector_store.search(FACT_COLLECTION, qvec, pool_n, filter={"session_id": session_id})

        # 4. BM25 keyword search (same session corpus — chunks + facts share
        #    one BM25 index keyed by session_id).
        bm25_hits = self.bm25.search(session_id, query, pool_n)
        bm25_scores = dict(bm25_hits)

        # 5. Merge/dedupe into one candidate pool, apply decay-adjusted score.
        merged: dict[str, Candidate] = {}

        def _add(id_: str, text: str, source: str, cos_score: float, anchor_turn: int, usage_count: int):
            bm = bm25_scores.get(id_, 0.0)
            score = cos_score + 0.1 * bm  # simple linear fusion of cosine + bm25
            if enable_decay:
                score *= decay_weight(anchor_turn, current_turn, usage_count, self.settings.decay_half_life_turns)
            if id_ not in merged or score > merged[id_].prelim_score:
                merged[id_] = Candidate(id=id_, text=text, source=source, prelim_score=score)

        for h in chunk_hits:
            _add(
                h.id,
                h.payload.get("text", ""),
                "tier2",
                h.score,
                h.payload.get("anchor_turn", 0),
                h.payload.get("usage_count", 0),
            )
        for h in fact_hits:
            _add(
                h.id,
                h.payload.get("text", ""),
                "tier3",
                h.score,
                h.payload.get("anchor_turn", current_turn),
                h.payload.get("usage_count", 0),
            )

        pool = sorted(merged.values(), key=lambda c: c.prelim_score, reverse=True)[: self.settings.retrieval_pool_size]

        # 6. LLM reranker picks the final top-k.
        reranked = self.reranker.rerank(query, pool, self.settings.retrieval_top_k)
        return reranked, hard_facts

    # ------------------------------------------------------------------
    def _assemble(self, wm_text: str, reranked: list[Candidate], hard_facts: list[dict]) -> str:
        parts = ["=== TIER 1: WORKING MEMORY (full) ===", wm_text or "(empty)"]
        if reranked:
            parts.append("\n=== RETRIEVED MEMORY (hybrid + reranked) ===")
            for c in reranked:
                parts.append(f"[mem:{c.id}] ({c.source}) {c.text}")
        if hard_facts:
            parts.append("\n=== HARD FACTS (deterministic entity match, Tier 3) ===")
            for f in hard_facts:
                parts.append(f"[mem:{f.get('id')}] ({f.get('category')}) {f}")
        return "\n".join(parts)

    def context_token_count(self, context_text: str) -> int:
        return count_tokens(context_text)
