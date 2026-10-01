"""Orchestrator Agent — the only user-facing agent. Calls the retrieval
pipeline, assembles context, generates the response, runs the (bounded)
Verifier pass, and dispatches the Memory Manager onto the background
executor (fire-and-forget — never blocks the response)."""
from __future__ import annotations

from dataclasses import dataclass

from tams.agents.memory_manager import MemoryManager
from tams.agents.verifier import VerificationReport, VerifierAgent
from tams.extractors.numeric import _NUMERIC_RE
from tams.llm.client import ChatResult
from tams.retrieval.hybrid import RetrievalResult
from tams.session import Session
from tams.storage.working_memory import Turn

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a tiered memory system. "
    "Ground every factual claim in the MEMORY CONTEXT provided below the "
    "system prompt; do not state numbers, dates, decisions, or constraints "
    "that are not present there. Prefer the most recent fact when memory "
    "contains conflicting versions, unless the user clearly intends an "
    "earlier one."
)

_REGEN_INSTRUCTION = (
    "\n\nIMPORTANT: Your previous answer contained claims that could not be "
    "grounded in the memory context. Regenerate your answer, stating ONLY "
    "facts that are explicitly present in the MEMORY CONTEXT. If you are "
    "unsure, say so rather than inventing a value."
)


@dataclass
class OrchestratorResult:
    response_text: str
    tagged_text: str
    verification: VerificationReport
    retrieval: RetrievalResult
    chat_result: ChatResult
    regenerated: bool
    turn_id: int
    cost_report_hint: str = "see tams.storage.cost_ledger.report()"


class OrchestratorAgent:
    def __init__(self, session: Session, async_memory: bool = True):
        self.session = session
        self.async_memory = async_memory
        self.memory_manager = MemoryManager(session)
        self.verifier = VerifierAgent(session.llm_client, session.settings)

    def handle_turn(self, user_message: str, system_prompt: str | None = None) -> OrchestratorResult:
        session = self.session
        session.turn_counter += 1
        turn_id = session.turn_counter

        user_turn = Turn(turn_id=turn_id, session_id=session.session_id, speaker="user", content=user_message)
        session.working_memory.append(user_turn)

        retrieval = session.retriever.retrieve(session.session_id, user_message, turn_id, session.fact_store)

        base_messages = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "system", "content": "MEMORY CONTEXT:\n" + retrieval.context_text},
            {"role": "user", "content": user_message},
        ]
        chat_result = session.llm_client.chat(
            base_messages,
            model=session.settings.orchestrator_model,
            purpose="orchestrator",
            fake_fn=lambda _m: self._fake_response(user_message, retrieval),
        )
        response_text = chat_result.text

        active_constraints = session.fact_store.fact_store.active_facts("constraint")
        verification = self.verifier.verify(response_text, retrieval.context_text, active_constraints)

        regenerated = False
        if verification.hallucination_rate > session.settings.hallucination_regen_threshold:
            # Bounded: exactly one regeneration attempt, never a loop.
            regen_messages = base_messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": _REGEN_INSTRUCTION},
            ]
            chat_result = session.llm_client.chat(
                regen_messages,
                model=session.settings.orchestrator_model,
                purpose="orchestrator_regen",
                fake_fn=lambda _m: self._fake_response(user_message, retrieval, strict=True),
            )
            response_text = chat_result.text
            verification = self.verifier.verify(response_text, retrieval.context_text, active_constraints)
            regenerated = True

        for f in retrieval.hard_facts:
            session.fact_store.mark_used(f["category"], f["id"], turn_id)

        assistant_turn = Turn(
            turn_id=turn_id, session_id=session.session_id, speaker="assistant", content=response_text
        )
        session.working_memory.append(assistant_turn)

        context_turns = session.working_memory.get_all(session.session_id)[-8:]
        if self.async_memory:
            self.memory_manager.dispatch_async(user_turn, context_turns)
            self.memory_manager.dispatch_async(assistant_turn, context_turns)
        else:
            self.memory_manager.process_turn(user_turn, context_turns)
            self.memory_manager.process_turn(assistant_turn, context_turns)

        return OrchestratorResult(
            response_text=response_text,
            tagged_text=verification.tagged_text,
            verification=verification,
            retrieval=retrieval,
            chat_result=chat_result,
            regenerated=regenerated,
            turn_id=turn_id,
        )

    @staticmethod
    def _fake_response(user_message: str, retrieval: RetrievalResult, strict: bool = False) -> str:
        """Deterministic fake-mode response: cites hard facts verbatim so the
        verifier/eval harness has something real to check against, without
        requiring an API key. Deliberately does NOT echo the raw user_message
        (that would re-inject any numbers/facts the user just stated back
        into the assistant turn, double-counting them at extraction time)."""
        # Tier 1 (full, current turn included) always wins over a possibly
        # one-turn-stale Tier 3 fact — the extractor for *this* turn hasn't
        # run yet (it runs post-response, async). A real LLM would read the
        # raw current message directly rather than only trusting retrieval;
        # this stub mimics that by preferring any value restated inline.
        inline_values = {m.group(1).lower(): m.group(2) for m in _NUMERIC_RE.finditer(user_message)}

        bits = []
        for f in retrieval.hard_facts[:3]:
            cat = f.get("category")
            if cat == "numeric":
                value = inline_values.get(str(f.get("param", "")).lower(), f.get("value"))
                bits.append(f"{f.get('param')} is {value} {f.get('unit') or ''}".strip() + ".")
            elif cat == "constraint":
                bits.append(f"Constraint: {f.get('constraint_text')}")
            elif cat == "decision":
                bits.append(f"Decision: {f.get('decision')}")
            elif cat == "dependency":
                bits.append(f"{f.get('entity')} was last modified at turn {f.get('modified_at_turn')}.")
            elif cat == "preference":
                bits.append(f"Preference noted: {f.get('pref_type')} = {f.get('value')}.")
        if not bits:
            bits.append("I don't have a specific stored fact for that yet.")
        return " ".join(bits)
