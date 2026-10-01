"""BM25 keyword index — the lexical half of hybrid retrieval. In-process
(rank_bm25) over the same corpus of chunk/fact text that's embedded into the
vector store, so no separate search-engine service is required."""
from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """One index per session. Rebuilt lazily (cheap: pure-python tokenizer +
    BM25Okapi over a few thousand docs is sub-millisecond)."""

    def __init__(self):
        self._docs: dict[str, list[tuple[str, str]]] = {}  # session_id -> [(doc_id, text)]
        self._lock = threading.Lock()

    def add(self, session_id: str, doc_id: str, text: str) -> None:
        with self._lock:
            self._docs.setdefault(session_id, []).append((doc_id, text))

    def remove(self, session_id: str, doc_id: str) -> None:
        with self._lock:
            docs = self._docs.get(session_id, [])
            self._docs[session_id] = [d for d in docs if d[0] != doc_id]

    def search(self, session_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
        with self._lock:
            docs = list(self._docs.get(session_id, []))
        if not docs:
            return []
        corpus = [_tokenize(text) for _, text in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(zip((d[0] for d in docs), scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
