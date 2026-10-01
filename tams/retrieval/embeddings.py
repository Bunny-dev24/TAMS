"""Query embedding cache — latency-budget guardrail: hybrid retrieval +
reranker already adds 200-500ms, so the embedding call for repeated/similar
queries should be cached rather than re-paid every turn. Redis-backed if
available, in-process LRU-ish dict otherwise (graceful degradation)."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

from tams.llm.client import LLMClient


def _hash_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode("utf-8")).hexdigest()


class InMemoryEmbeddingCache:
    def __init__(self, max_size: int = 5000):
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def set(self, key: str, value: list[float]) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)


class RedisEmbeddingCache:
    def __init__(self, url: str, ttl_s: int = 3600):
        import redis

        self._r = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_s
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[float] | None:
        raw = self._r.get(f"tams:embcache:{key}")
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: list[float]) -> None:
        self._r.set(f"tams:embcache:{key}", json.dumps(value), ex=self._ttl)

    def healthy(self) -> bool:
        try:
            self._r.ping()
            return True
        except Exception:
            return False


def build_embedding_cache(url: str):
    try:
        cache = RedisEmbeddingCache(url)
        if cache.healthy():
            return cache
    except Exception:
        pass
    return InMemoryEmbeddingCache()


class CachedEmbedder:
    """Wraps LLMClient.embed with a cache specifically for single-query
    embedding (the hot-path call); bulk chunk/fact embedding at write time
    bypasses the cache since those texts are rarely repeated."""

    def __init__(self, llm_client: LLMClient, cache=None):
        self.llm_client = llm_client
        self.cache = cache or InMemoryEmbeddingCache()

    def embed_query(self, text: str, model: str) -> list[float]:
        key = _hash_key(model, text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vec = self.llm_client.embed([text], purpose="query_embed")[0]
        self.cache.set(key, vec)
        return vec
