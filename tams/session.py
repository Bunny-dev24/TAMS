"""Wires every component together for a given session_id — the single place
that constructs Tier 1/2/3 stores, the retrieval pipeline, and the 3 agents,
honoring graceful-degradation (store construction never raises)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from tams.config import Settings
from tams.llm.client import LLMClient, get_llm_client
from tams.retrieval.embeddings import CachedEmbedder, build_embedding_cache
from tams.retrieval.hybrid import HybridRetriever
from tams.retrieval.reranker import LLMReranker
from tams.storage.bm25_index import BM25Index
from tams.storage.cost_ledger import make_cost_sink
from tams.storage.vector_store import VectorStore, build_vector_store
from tams.storage.working_memory import WorkingMemoryStore, build_working_memory_store
from tams.tiers.tier1_working_memory import WorkingMemory
from tams.tiers.tier2_chunk_store import ChunkStore
from tams.tiers.tier3_fact_store import Tier3FactStore

# Process-wide singletons shared across sessions (vector store / bm25 index /
# redis connections are safe + desirable to share; cheap to construct once).
_shared_vector_store: VectorStore | None = None
_shared_bm25: BM25Index | None = None
_shared_wm_store: WorkingMemoryStore | None = None
_shared_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def _get_shared(settings: Settings):
    global _shared_vector_store, _shared_bm25, _shared_wm_store, _shared_executor
    with _lock:
        if _shared_vector_store is None:
            _shared_vector_store = build_vector_store(settings.qdrant_url, settings.embedding_dim)
        if _shared_bm25 is None:
            _shared_bm25 = BM25Index()
        if _shared_wm_store is None:
            _shared_wm_store = build_working_memory_store(settings.redis_url)
        if _shared_executor is None:
            _shared_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tams-memory-manager")
    return _shared_vector_store, _shared_bm25, _shared_wm_store, _shared_executor


def reset_shared_state() -> None:
    """For tests: force fresh in-memory stores between test cases."""
    global _shared_vector_store, _shared_bm25, _shared_wm_store, _shared_executor
    with _lock:
        _shared_vector_store = None
        _shared_bm25 = None
        _shared_wm_store = None
        _shared_executor = None


@dataclass
class Session:
    session_id: str
    user_id: str
    settings: Settings
    llm_client: LLMClient
    vector_store: VectorStore
    bm25: BM25Index
    working_memory: WorkingMemory
    chunk_store: ChunkStore
    fact_store: Tier3FactStore
    retriever: HybridRetriever
    executor: ThreadPoolExecutor
    turn_counter: int = 0
    turns_since_last_preference_run: int = 0


def build_session(session_id: str, settings: Settings, user_id: str = "default") -> Session:
    vector_store, bm25, wm_store, executor = _get_shared(settings)
    cost_sink = make_cost_sink(session_id, settings)
    llm_client = LLMClient(settings, cost_sink=cost_sink)

    working_memory = WorkingMemory(wm_store, settings)
    chunk_store = ChunkStore(vector_store, bm25, llm_client, settings)
    fact_store = Tier3FactStore(session_id, vector_store, bm25, llm_client, user_id=user_id)

    embed_cache = build_embedding_cache(settings.redis_url)
    embedder = CachedEmbedder(llm_client, embed_cache)
    reranker = LLMReranker(llm_client, settings)
    retriever = HybridRetriever(working_memory, vector_store, bm25, embedder, reranker, settings)

    return Session(
        session_id=session_id,
        user_id=user_id,
        settings=settings,
        llm_client=llm_client,
        vector_store=vector_store,
        bm25=bm25,
        working_memory=working_memory,
        chunk_store=chunk_store,
        fact_store=fact_store,
        retriever=retriever,
        executor=executor,
    )
