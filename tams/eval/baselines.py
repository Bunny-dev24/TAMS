"""Baselines for the eval harness, per the project's ablation plan:
  * full-context stuffing (no retrieval at all)
  * plain cosine-only RAG (no BM25, no rerank, no deterministic entity lookup)
  * MemGPT-style pure narrative summarization (lossy, regenerated, replaces
    raw storage entirely — the thing TAMS v2 is explicitly designed to beat)
"""
from __future__ import annotations

from tams.config import Settings
from tams.llm.client import LLMClient
from tams.storage.vector_store import VectorStore
from tams.storage.working_memory import Turn
from tams.tiers.tier2_chunk_store import CHUNK_COLLECTION
from tams.tiers.tier3_fact_store import FACT_COLLECTION


def full_context_stuffing(all_turns: list[Turn]) -> str:
    return "\n".join(f"{t.speaker} (turn {t.turn_id}): {t.content}" for t in all_turns)


def cosine_only_rag(
    vector_store: VectorStore,
    embedder,
    settings: Settings,
    session_id: str,
    query: str,
    top_k: int,
) -> str:
    """No BM25, no LLM rerank, no deterministic entity lookup — purely
    top-k cosine similarity over whatever got embedded. The fairest 'plain
    RAG' comparison point."""
    qvec = embedder.embed_query(query, settings.embedding_model)
    chunk_hits = vector_store.search(CHUNK_COLLECTION, qvec, top_k, filter={"session_id": session_id})
    fact_hits = vector_store.search(FACT_COLLECTION, qvec, top_k, filter={"session_id": session_id})
    hits = sorted(chunk_hits + fact_hits, key=lambda h: h.score, reverse=True)[:top_k]
    return "\n".join(h.payload.get("text", "") for h in hits)


def memgpt_style_compress(llm_client: LLMClient, settings: Settings, turns: list[Turn]) -> str:
    """Lossy, REGENERATE-style narrative summarization of a turn span — the
    pattern TAMS v2 deliberately avoids for primary long-session memory.
    Each call compresses independently (no anchoring/extension), which is
    exactly what causes drift/loss over many compactions."""
    text = "\n".join(f"{t.speaker}: {t.content}" for t in turns)

    def _fake_compress(_msgs):
        words = text.split()
        return " ".join(words[: max(20, len(words) // 10)]) + " ... [compressed, detail lost]"

    result = llm_client.chat(
        [
            {
                "role": "system",
                "content": "Summarize this conversation span narratively and concisely. Some detail loss is acceptable.",
            },
            {"role": "user", "content": text},
        ],
        model=settings.extractor_model,
        purpose="baseline_memgpt_summary",
        fake_fn=_fake_compress,
    )
    return result.text
