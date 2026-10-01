"""Vector store abstraction. Real backend: Qdrant (docker-compose). Fallback:
in-process numpy cosine index — used automatically if Qdrant is unreachable
(graceful degradation guardrail) and in tests."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, collection: str, id: str, vector: list[float], payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(
        self, collection: str, vector: list[float], top_k: int, filter: dict[str, Any] | None = None
    ) -> list[VectorHit]: ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None: ...

    @abstractmethod
    def get_all(self, collection: str, filter: dict[str, Any] | None = None) -> list[VectorHit]: ...

    @abstractmethod
    def healthy(self) -> bool: ...


class InMemoryVectorStore(VectorStore):
    """Numpy cosine-similarity index. No external service required."""

    def __init__(self):
        self._data: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def _coll(self, collection: str) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
        with self._lock:
            return self._data.setdefault(collection, {})

    def upsert(self, collection: str, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self._coll(collection)[id] = (v, payload)

    def _match(self, payload: dict[str, Any], filter: dict[str, Any] | None) -> bool:
        if not filter:
            return True
        return all(payload.get(k) == v for k, v in filter.items())

    def search(
        self, collection: str, vector: list[float], top_k: int, filter: dict[str, Any] | None = None
    ) -> list[VectorHit]:
        q = np.array(vector, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        hits = []
        for id, (v, payload) in self._coll(collection).items():
            if not self._match(payload, filter):
                continue
            score = float(np.dot(q, v))
            hits.append(VectorHit(id=id, score=score, payload=payload))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete(self, collection: str, ids: list[str]) -> None:
        coll = self._coll(collection)
        for id in ids:
            coll.pop(id, None)

    def get_all(self, collection: str, filter: dict[str, Any] | None = None) -> list[VectorHit]:
        return [
            VectorHit(id=id, score=1.0, payload=payload)
            for id, (_, payload) in self._coll(collection).items()
            if self._match(payload, filter)
        ]

    def healthy(self) -> bool:
        return True


class QdrantVectorStore(VectorStore):
    """Production backend talking to a Qdrant instance (see docker-compose.yml)."""

    def __init__(self, url: str, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._dim = dim
        self._ensured: set[str] = set()

    def _ensure_collection(self, collection: str) -> None:
        if collection in self._ensured:
            return
        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=self._qmodels.VectorParams(
                    size=self._dim, distance=self._qmodels.Distance.COSINE
                ),
            )
        self._ensured.add(collection)

    def upsert(self, collection: str, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self._ensure_collection(collection)
        self._client.upsert(
            collection_name=collection,
            points=[self._qmodels.PointStruct(id=id, vector=vector, payload=payload)],
        )

    def search(
        self, collection: str, vector: list[float], top_k: int, filter: dict[str, Any] | None = None
    ) -> list[VectorHit]:
        self._ensure_collection(collection)
        qfilter = None
        if filter:
            qfilter = self._qmodels.Filter(
                must=[
                    self._qmodels.FieldCondition(key=k, match=self._qmodels.MatchValue(value=v))
                    for k, v in filter.items()
                ]
            )
        res = self._client.search(
            collection_name=collection, query_vector=vector, limit=top_k, query_filter=qfilter
        )
        return [VectorHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in res]

    def delete(self, collection: str, ids: list[str]) -> None:
        self._ensure_collection(collection)
        self._client.delete(collection_name=collection, points_selector=self._qmodels.PointIdsList(points=ids))

    def get_all(self, collection: str, filter: dict[str, Any] | None = None) -> list[VectorHit]:
        self._ensure_collection(collection)
        points, _ = self._client.scroll(collection_name=collection, limit=10_000)
        hits = [VectorHit(id=str(p.id), score=1.0, payload=p.payload or {}) for p in points]
        if filter:
            hits = [h for h in hits if all(h.payload.get(k) == v for k, v in filter.items())]
        return hits

    def healthy(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


def build_vector_store(url: str, dim: int) -> VectorStore:
    """Try Qdrant; fall back to in-memory on any connection problem
    (graceful degradation guardrail #7)."""
    try:
        store = QdrantVectorStore(url=url, dim=dim)
        if store.healthy():
            return store
    except Exception:
        pass
    return InMemoryVectorStore()
