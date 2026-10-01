from tams.llm.client import LLMClient
from tams.retrieval.embeddings import CachedEmbedder, InMemoryEmbeddingCache
from tams.retrieval.hybrid import HybridRetriever
from tams.retrieval.reranker import LLMReranker
from tams.storage.bm25_index import BM25Index
from tams.storage.vector_store import InMemoryVectorStore
from tams.storage.working_memory import InMemoryWorkingMemoryStore, Turn
from tams.tiers.tier1_working_memory import WorkingMemory
from tams.tiers.tier2_chunk_store import ChunkStore
from tams.tiers.tier3_fact_store import Tier3FactStore


def _build(settings):
    llm = LLMClient(settings)
    vs = InMemoryVectorStore()
    bm25 = BM25Index()
    wm = WorkingMemory(InMemoryWorkingMemoryStore(), settings)
    chunk_store = ChunkStore(vs, bm25, llm, settings)
    fact_store = Tier3FactStore("s1", vs, bm25, llm)
    embedder = CachedEmbedder(llm, InMemoryEmbeddingCache())
    reranker = LLMReranker(llm, settings)
    retriever = HybridRetriever(wm, vs, bm25, embedder, reranker, settings)
    return llm, wm, chunk_store, fact_store, retriever


def test_tier1_always_fully_included(settings):
    llm, wm, chunk_store, fact_store, retriever = _build(settings)
    wm.append(Turn(turn_id=1, session_id="s1", speaker="user", content="unique_wm_marker_xyz"))
    result = retriever.retrieve("s1", "irrelevant query", current_turn=1, fact_store=fact_store)
    assert "unique_wm_marker_xyz" in result.context_text
    assert not result.degraded


def test_hard_facts_included_via_entity_match_regardless_of_rank(settings):
    llm, wm, chunk_store, fact_store, retriever = _build(settings)
    fact_store.write("dependency", turn_id=1, confidence=0.8, entity="SpecialEntityZZZ", modified_at_turn=1, referenced_at_turns=[])
    result = retriever.retrieve("s1", "What happened to SpecialEntityZZZ?", current_turn=5, fact_store=fact_store)
    assert len(result.hard_facts) == 1
    assert result.hard_facts[0]["entity"] == "SpecialEntityZZZ"


def test_chunk_retrieval_surfaces_relevant_chunk(settings):
    llm, wm, chunk_store, fact_store, retriever = _build(settings)
    turns = [
        Turn(turn_id=i, session_id="s1", speaker="user", content=f"filler topic number {i}")
        for i in range(1, 4)
    ]
    turns.append(Turn(turn_id=4, session_id="s1", speaker="user", content="the quantum flux capacitor needs calibration"))
    chunk_store.ingest_evicted_turns("s1", turns)

    result = retriever.retrieve("s1", "quantum flux capacitor calibration", current_turn=10, fact_store=fact_store)
    assert any("quantum flux capacitor" in c.text for c in result.reranked_items)


def test_graceful_degradation_on_pipeline_failure(settings, monkeypatch):
    llm, wm, chunk_store, fact_store, retriever = _build(settings)
    wm.append(Turn(turn_id=1, session_id="s1", speaker="user", content="still here"))

    def _boom(*args, **kwargs):
        raise RuntimeError("vector store exploded")

    monkeypatch.setattr(retriever.vector_store, "search", _boom)
    result = retriever.retrieve("s1", "anything", current_turn=1, fact_store=fact_store)
    assert result.degraded is True
    assert "still here" in result.context_text  # Tier 1 fallback still works


def test_ablation_no_hybrid_rerank_skips_bm25_and_entity_lookup_scoring(settings):
    llm, wm, chunk_store, fact_store, retriever = _build(settings)
    retriever.enable_hybrid_rerank = False
    turns = [Turn(turn_id=1, session_id="s1", speaker="user", content="some content about widgets")]
    chunk_store.ingest_evicted_turns("s1", turns)
    result = retriever.retrieve("s1", "widgets", current_turn=1, fact_store=fact_store)
    assert not result.degraded
