"""Tier 3 wrapper: structured fact CRUD (tams.storage.fact_db.FactStore) PLUS
embedding/BM25 indexing of fact text, so the hybrid retrieval pipeline can
similarity-search Tier 3 facts exactly like Tier 2 chunks, in addition to the
deterministic entity-match lookup."""
from __future__ import annotations

from typing import Any

from tams.llm.client import LLMClient
from tams.storage.bm25_index import BM25Index
from tams.storage.fact_db import FactStore, FactWriteResult
from tams.storage.vector_store import VectorStore

FACT_COLLECTION = "tier3_facts"


def _fact_text(category: str, fields: dict[str, Any]) -> str:
    if category == "numeric":
        return f"{fields.get('param')} = {fields.get('value')} {fields.get('unit') or ''}".strip()
    if category == "constraint":
        return f"CONSTRAINT ({fields.get('scope')}): {fields.get('constraint_text')} — reason: {fields.get('reason')}"
    if category == "decision":
        return (
            f"DECISION: {fields.get('decision')} — rejected: "
            f"{', '.join(fields.get('rejected_alternatives') or [])} — reason: {fields.get('reason')}"
        )
    if category == "dependency":
        return f"ENTITY {fields.get('entity')} modified at turn {fields.get('modified_at_turn')}"
    if category == "preference":
        return f"PREFERENCE {fields.get('pref_type')}: {fields.get('value')}"
    return str(fields)


class Tier3FactStore:
    def __init__(
        self,
        session_id: str,
        vector_store: VectorStore,
        bm25: BM25Index,
        llm_client: LLMClient,
        user_id: str = "default",
    ):
        self.fact_store = FactStore(session_id=session_id, user_id=user_id)
        self.vector_store = vector_store
        self.bm25 = bm25
        self.llm_client = llm_client
        self.session_id = session_id

    def write(self, category: str, turn_id: int, confidence: float, **fields) -> FactWriteResult:
        result = self.fact_store.write(category, turn_id, confidence, **fields)
        text = _fact_text(category, fields)
        vec = self.llm_client.embed([text], purpose=f"tier3_fact_embed:{category}")[0]
        self.vector_store.upsert(
            FACT_COLLECTION,
            result.fact_id,
            vec,
            {"session_id": self.session_id, "category": category, "fact_id": result.fact_id, "text": text},
        )
        self.bm25.add(self.session_id, result.fact_id, text)
        if result.conflicted_with:
            self.vector_store.delete(FACT_COLLECTION, [result.conflicted_with])
            self.bm25.remove(self.session_id, result.conflicted_with)
        return result

    def match_entities(self, entities: list[str]) -> list[dict[str, Any]]:
        out = []
        for category in ("numeric", "constraint", "decision", "dependency", "preference"):
            out.extend(self.fact_store.match_entities(category, entities))
        return out

    def all_active(self) -> list[dict[str, Any]]:
        return self.fact_store.all_active_across_categories()

    def mark_used(self, category: str, fact_id: str, turn_id: int) -> None:
        self.fact_store.mark_used(category, fact_id, turn_id)

    def record_dependency_reference(self, entity: str, turn_id: int) -> list[str]:
        return self.fact_store.record_dependency_reference(entity, turn_id)
