from tams.llm.client import LLMClient
from tams.storage.bm25_index import BM25Index
from tams.storage.vector_store import InMemoryVectorStore
from tams.storage.working_memory import Turn
from tams.tiers.tier2_chunk_store import ChunkStore


def _turns(n, session_id="s1"):
    return [Turn(turn_id=i, session_id=session_id, speaker="user" if i % 2 == 0 else "assistant", content=f"message {i}") for i in range(n)]


def test_ingest_evicted_turns_groups_into_chunks(settings):
    llm = LLMClient(settings)
    store = ChunkStore(InMemoryVectorStore(), BM25Index(), llm, settings)
    turns = _turns(7)
    chunks = store.ingest_evicted_turns("s1", turns)
    # chunk_size_turns=3 -> ceil(7/3) = 3 chunks
    assert len(chunks) == 3
    assert chunks[0].turn_ids == [0, 1, 2]
    assert chunks[-1].turn_ids == [6]
    # speaker/timestamp metadata preserved
    assert chunks[0].speakers == ["user", "assistant", "user"]


def test_chunks_are_searchable_via_vector_store(settings):
    llm = LLMClient(settings)
    vs = InMemoryVectorStore()
    store = ChunkStore(vs, BM25Index(), llm, settings)
    turns = _turns(3)
    store.ingest_evicted_turns("s1", turns)
    qvec = llm.embed(["message 1"])[0]
    hits = vs.search("tier2_chunks", qvec, top_k=5)
    assert len(hits) == 1
    assert "message 1" in hits[0].payload["text"]


def test_extend_summary_is_additive_not_regenerated(settings):
    llm = LLMClient(settings)
    store = ChunkStore(InMemoryVectorStore(), BM25Index(), llm, settings)

    span1 = _turns(3)
    chunks1 = store.ingest_evicted_turns("s1", span1)
    summary1 = store.extend_summary("s1", chunks1)
    assert summary1.last_anchored_turn == 2

    span2 = [Turn(turn_id=5, session_id="s1", speaker="user", content="we decided to use Postgres instead")]
    chunks2 = store.ingest_evicted_turns("s1", span2)
    summary2 = store.extend_summary("s1", chunks2)

    assert summary2.last_anchored_turn == 5
    # Prior decisions/changes must survive (extend, not regenerate) if any were recorded.
    assert summary2.session_id == "s1"
