"""Tier 2 — Raw Chunk Store: the PRIMARY long-session memory.

Replaces heavy narrative summarization. On eviction from Tier 1, turns are
grouped into fixed-size raw chunks (speaker + timestamp metadata preserved
verbatim) and embedded — zero *generation* cost at write time (embeddings
only, no LLM completion call).

Also maintains the lightweight "anchored incremental summary": a running
session-state document (intent, changes made, decisions, next-steps) that is
EXTENDED per eviction span, never regenerated from scratch.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select

from tams.config import Settings
from tams.llm.client import LLMClient
from tams.storage.bm25_index import BM25Index
from tams.storage.db import get_session
from tams.storage.models import SessionSummary
from tams.storage.vector_store import VectorStore
from tams.storage.working_memory import Turn

CHUNK_COLLECTION = "tier2_chunks"


@dataclass
class Chunk:
    chunk_id: str
    session_id: str
    turn_ids: list[int]
    speakers: list[str]
    timestamps: list[str]
    text: str
    anchor_turn: int  # last turn covered — used for decay aging


def _chunk_text(turns: list[Turn]) -> str:
    lines = [f"[{t.timestamp}] {t.speaker} (turn {t.turn_id}): {t.content}" for t in turns]
    return "\n".join(lines)


class ChunkStore:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Index,
        llm_client: LLMClient,
        settings: Settings,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.llm_client = llm_client
        self.settings = settings

    # ------------------------------------------------------------ write side
    def ingest_evicted_turns(self, session_id: str, turns: list[Turn]) -> list[Chunk]:
        """Group evicted turns into fixed-size chunks, embed, store. No LLM
        completion call — embeddings only (per spec: zero LLM cost at write)."""
        size = self.settings.chunk_size_turns
        chunks: list[Chunk] = []
        for i in range(0, len(turns), size):
            group = turns[i : i + size]
            if not group:
                continue
            text = _chunk_text(group)
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                session_id=session_id,
                turn_ids=[t.turn_id for t in group],
                speakers=[t.speaker for t in group],
                timestamps=[t.timestamp for t in group],
                text=text,
                anchor_turn=group[-1].turn_id,
            )
            chunks.append(chunk)

        if chunks:
            vectors = self.llm_client.embed([c.text for c in chunks], purpose="tier2_chunk_embed")
            for chunk, vec in zip(chunks, vectors):
                self.vector_store.upsert(
                    CHUNK_COLLECTION,
                    chunk.chunk_id,
                    vec,
                    {
                        "session_id": chunk.session_id,
                        "turn_ids": chunk.turn_ids,
                        "speakers": chunk.speakers,
                        "timestamps": chunk.timestamps,
                        "text": chunk.text,
                        "anchor_turn": chunk.anchor_turn,
                    },
                )
                self.bm25.add(chunk.session_id, chunk.chunk_id, chunk.text)
        return chunks

    # ----------------------------------------------------- anchored summary
    def extend_summary(self, session_id: str, new_chunks: list[Chunk], user_id: str = "default") -> SessionSummary:
        """Extend (never regenerate) the running session-state doc with the
        content of the newly-evicted span."""
        if not new_chunks:
            with get_session() as db:
                existing = db.get(SessionSummary, session_id)
                return existing or SessionSummary(session_id=session_id)

        span_text = "\n\n".join(c.text for c in new_chunks)
        with get_session() as db:
            prior = db.get(SessionSummary, session_id)
            prior_doc = {
                "intent": prior.intent if prior else "",
                "changes_made": prior.changes_made if prior else [],
                "decisions": prior.decisions if prior else [],
                "next_steps": prior.next_steps if prior else [],
            }

        def _fake_extend(messages):
            # Deterministic heuristic extension used when no LLM key is set.
            changes = list(prior_doc["changes_made"])
            decisions = list(prior_doc["decisions"])
            next_steps = list(prior_doc["next_steps"])
            for line in span_text.splitlines():
                low = line.lower()
                if "decided" in low or "we'll go with" in low or "chosen" in low:
                    decisions.append(line.strip())
                elif "changed" in low or "updated" in low or "implemented" in low:
                    changes.append(line.strip())
                elif "next" in low or "todo" in low or "will " in low:
                    next_steps.append(line.strip())
            return {
                "intent": prior_doc["intent"] or (span_text.splitlines()[0][:200] if span_text else ""),
                "changes_made": changes[-50:],
                "decisions": decisions[-50:],
                "next_steps": next_steps[-20:],
            }

        schema = {
            "intent": "string",
            "changes_made": "string[]",
            "decisions": "string[]",
            "next_steps": "string[]",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You maintain an ANCHORED, EXTEND-ONLY session-state summary. "
                    "You are given the prior summary document and a NEW span of raw "
                    "conversation chunks. Extend the document — do not regenerate it "
                    "from scratch, do not drop prior entries unless clearly obsolete."
                ),
            },
            {
                "role": "user",
                "content": f"PRIOR SUMMARY:\n{prior_doc}\n\nNEW SPAN:\n{span_text}",
            },
        ]
        extended = self.llm_client.chat_json(
            messages,
            model=self.settings.extractor_model,
            schema=schema,
            purpose="tier2_anchored_summary",
            fake_fn=_fake_extend,
        )

        with get_session() as db:
            row = db.get(SessionSummary, session_id)
            if row is None:
                row = SessionSummary(session_id=session_id, user_id=user_id)
                db.add(row)
            row.intent = extended.get("intent", row.intent)
            row.changes_made = extended.get("changes_made", row.changes_made)
            row.decisions = extended.get("decisions", row.decisions)
            row.next_steps = extended.get("next_steps", row.next_steps)
            row.last_anchored_turn = max(c.anchor_turn for c in new_chunks)
            db.flush()
            return row

    def get_summary(self, session_id: str) -> SessionSummary | None:
        with get_session() as db:
            return db.get(SessionSummary, session_id)

    # ------------------------------------------------------------- privacy
    def delete_session(self, session_id: str) -> None:
        hits = self.vector_store.get_all(CHUNK_COLLECTION, filter={"session_id": session_id})
        ids = [h.id for h in hits]
        if ids:
            self.vector_store.delete(CHUNK_COLLECTION, ids)
        for id in ids:
            self.bm25.remove(session_id, id)
        with get_session() as db:
            row = db.get(SessionSummary, session_id)
            if row:
                db.delete(row)
