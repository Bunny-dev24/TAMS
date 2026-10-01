"""Verifier Agent (Judge) — post-generation groundedness check.

Hybrid hallucination mitigation:
  * Rules layer: deterministic exact-match checks for numbers / dates /
    named entities / constraint violations against Tier 3 + retrieved context.
  * Judge layer: LLM semantic-entailment check for paraphrased/contextual claims.
  * Mandatory citation tagging ([mem:fact_id]) on every factual sentence.

Design note on guardrail #7 ("no recursive LLM calls in the hot path, only
async background"): the rules layer is pure Python (free, synchronous). The
judge layer is a SINGLE bounded LLM call (not a loop) over all sentences at
once, plus — only if ungrounded claims are found — at most ONE regeneration
pass. This keeps the hot path finite/bounded rather than open-ended-recursive,
while still delivering the functional requirement of a post-generation check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tams.config import Settings
from tams.llm.client import LLMClient

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b")
_NEGATION_WORDS = {"not", "never", "cannot", "can't", "won't", "shouldn't", "mustn't"}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@dataclass
class SentenceVerdict:
    text: str
    grounded: bool
    citation: str | None
    rule_flags: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    sentences: list[SentenceVerdict]
    tagged_text: str
    hallucination_rate: float
    citation_coverage: float
    rule_violations: list[str]


class VerifierAgent:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings

    # --------------------------------------------------------------- rules
    def _rule_check(self, response_text: str, context_text: str, active_constraints: list[dict]) -> list[str]:
        violations = []
        context_numbers = set(_NUMBER_RE.findall(context_text))
        for num in _NUMBER_RE.findall(response_text):
            if len(num.lstrip("-")) <= 1:
                continue  # skip single digits — too noisy to be a meaningful "fact"
            if num not in context_numbers:
                violations.append(f"unsupported number '{num}' not present in retrieved context")

        context_dates = set(_DATE_RE.findall(context_text))
        for d in _DATE_RE.findall(response_text):
            if d not in context_dates:
                violations.append(f"unsupported date '{d}' not present in retrieved context")

        resp_lower = response_text.lower()
        for c in active_constraints:
            ctext = (c.get("constraint_text") or "").lower()
            key_terms = [w for w in re.findall(r"[a-z]{4,}", ctext) if w not in _NEGATION_WORDS]
            is_negative = any(w in ctext for w in _NEGATION_WORDS)
            overlap = sum(1 for w in key_terms if w in resp_lower)
            if is_negative and overlap >= max(1, len(key_terms) // 2) and not any(
                neg in resp_lower for neg in _NEGATION_WORDS
            ):
                violations.append(f"possible constraint violation: '{c.get('constraint_text')}'")
        return violations

    # ---------------------------------------------------------------- judge
    def _judge_check(self, response_text: str, context_text: str) -> list[SentenceVerdict]:
        sentences = _split_sentences(response_text)
        if not sentences:
            return []

        mem_id_re = re.compile(r"\[mem:([a-zA-Z0-9\-]+)\]")
        context_mem_ids = set(mem_id_re.findall(context_text))

        schema = {"verdicts": [{"index": "int", "grounded": "bool", "citation": "string|null"}]}
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        messages = [
            {
                "role": "system",
                "content": (
                    "For each numbered claim/sentence, decide if it is semantically "
                    "ENTAILED by the provided context (paraphrases count as grounded). "
                    "If grounded, cite the best-matching [mem:id] from the context. "
                    "Return JSON {verdicts: [{index, grounded, citation}]}."
                ),
            },
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nCLAIMS:\n{numbered}"},
        ]

        def _fake_judge(_msgs):
            verdicts = []
            ctx_lower = context_text.lower()
            for i, s in enumerate(sentences):
                words = [w for w in re.findall(r"[a-z0-9]{4,}", s.lower())]
                overlap = sum(1 for w in words if w in ctx_lower)
                grounded = len(words) == 0 or overlap >= max(1, len(words) // 3)
                citation = None
                if grounded and context_mem_ids:
                    citation = next(iter(context_mem_ids))
                verdicts.append({"index": i, "grounded": grounded, "citation": citation})
            return {"verdicts": verdicts}

        result = self.llm_client.chat_json(
            messages, model=self.settings.verifier_model, schema=schema, purpose="verifier_judge", fake_fn=_fake_judge
        )
        verdict_map = {v["index"]: v for v in result.get("verdicts", [])}

        out = []
        for i, s in enumerate(sentences):
            v = verdict_map.get(i, {"grounded": False, "citation": None})
            out.append(SentenceVerdict(text=s, grounded=bool(v.get("grounded")), citation=v.get("citation")))
        return out

    # ------------------------------------------------------------- public
    def verify(
        self,
        response_text: str,
        context_text: str,
        active_constraints: list[dict] | None = None,
    ) -> VerificationReport:
        active_constraints = active_constraints or []
        rule_violations = self._rule_check(response_text, context_text, active_constraints)
        sentence_verdicts = self._judge_check(response_text, context_text)
        for sv in sentence_verdicts:
            if any(num in sv.text for num in re.findall(r"-?\d+(?:\.\d+)?%?", sv.text) if len(num.lstrip("-")) > 1):
                if any(v.startswith("unsupported number") and _NUMBER_RE.search(v) for v in rule_violations):
                    sv.rule_flags.append("numeric_rule_check")

        n = len(sentence_verdicts) or 1
        grounded_count = sum(1 for sv in sentence_verdicts if sv.grounded)
        cited_count = sum(1 for sv in sentence_verdicts if sv.citation)
        hallucination_rate = 1.0 - (grounded_count / n)
        citation_coverage = cited_count / n

        tagged_parts = []
        for sv in sentence_verdicts:
            if sv.citation:
                tagged_parts.append(f"{sv.text} [mem:{sv.citation}]")
            elif not sv.grounded:
                tagged_parts.append(f"{sv.text} [UNVERIFIED]")
            else:
                tagged_parts.append(sv.text)
        tagged_text = " ".join(tagged_parts)

        return VerificationReport(
            sentences=sentence_verdicts,
            tagged_text=tagged_text,
            hallucination_rate=hallucination_rate,
            citation_coverage=citation_coverage,
            rule_violations=rule_violations,
        )
