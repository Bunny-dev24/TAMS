"""Synthetic needle-in-haystack generator across the 5 protected Tier-3
categories — the paper's core eval contribution (recall@k PER CATEGORY, not
just generic fact recall)."""
from __future__ import annotations

import random
from dataclasses import dataclass

_FILLER_TOPICS = [
    "the weather has been nice lately",
    "I watched a documentary about deep sea creatures",
    "let's talk about the UI color scheme for a bit",
    "I'm thinking about reorganizing my bookshelf",
    "the coffee machine in the office broke again",
    "there's a new restaurant opening downtown",
    "I need to renew my passport soon",
    "the traffic this morning was unusually light",
    "my plant finally bloomed after months",
    "I've been meaning to learn to play chess better",
]

NEEDLE_TEMPLATES = {
    "numeric": [
        ("request_timeout is set to {v} ms", "What is the request_timeout value?", "{v}"),
        ("max_retries is 5 and backoff_multiplier = 2.0", "What is backoff_multiplier?", "2.0"),
        ("the monthly_budget is set to {v} usd", "What is the monthly_budget?", "{v}"),
    ],
    "constraint": [
        (
            "The export job must never run during business hours because it locks the DB.",
            "What constraint applies to the export job?",
            "export job must never run during business hours",
        ),
        (
            "Users should not be allowed to delete the audit log; it is required for compliance.",
            "Can users delete the audit log?",
            "should not be allowed to delete the audit log",
        ),
    ],
    "decision": [
        (
            "We decided to go with PostgreSQL instead of MongoDB because we needed strong transactional guarantees.",
            "Which database was chosen and why?",
            "PostgreSQL",
        ),
        (
            "We chose server-sent events over websockets because our traffic is one-directional.",
            "What communication method was chosen over websockets?",
            "server-sent events",
        ),
    ],
    "dependency": [
        (
            "I renamed the AuthModule to IdentityModule to better reflect its scope.",
            "What was AuthModule renamed to?",
            "IdentityModule",
        ),
        (
            "I changed the PaymentGateway to use the new StripeAdapter.",
            "What does PaymentGateway now use?",
            "StripeAdapter",
        ),
    ],
    "preference": [
        ("Just give me the short answer, no explanation needed.", None, "concise"),
        ("tl;dr please, I don't need the full breakdown.", None, "concise"),
        ("I always use pytest for testing, never unittest.", None, "pytest"),
        ("I always use pytest for my test suite setup too.", None, "pytest"),
        ("I always use pytest in CI as well.", None, "pytest"),
    ],
}


@dataclass
class Needle:
    category: str
    turn_id: int
    content: str
    query: str | None
    expected_answer_fragment: str


@dataclass
class HaystackSpec:
    turns: list[tuple[str, str]]  # (speaker, content), speaker alternates user/assistant implicitly
    needles: list[Needle]


def generate_haystack(
    num_filler_turns: int = 200,
    seed: int = 42,
) -> HaystackSpec:
    rng = random.Random(seed)
    turns: list[tuple[str, str]] = []
    needles: list[Needle] = []

    all_needles = []
    for category, templates in NEEDLE_TEMPLATES.items():
        for template, query, answer in templates:
            all_needles.append((category, template, query, answer))
    rng.shuffle(all_needles)

    # Spread needles roughly evenly across the haystack.
    positions = sorted(rng.sample(range(5, num_filler_turns - 5), k=min(len(all_needles), num_filler_turns - 10)))

    needle_iter = iter(all_needles)
    position_set = set(positions)
    turn_id = 0
    for i in range(num_filler_turns):
        turn_id += 1
        if i in position_set:
            category, template, query, answer = next(needle_iter)
            value = str(rng.randint(100, 9000))
            content = template.format(v=value)
            answer_fragment = answer.format(v=value) if "{v}" in answer else answer
            speaker = "user"
            turns.append((speaker, content))
            needles.append(
                Needle(category=category, turn_id=turn_id, content=content, query=query, expected_answer_fragment=answer_fragment)
            )
        else:
            speaker = "user" if i % 2 == 0 else "assistant"
            content = rng.choice(_FILLER_TOPICS) + f" (filler turn {i})"
            turns.append((speaker, content))

    return HaystackSpec(turns=turns, needles=[n for n in needles if n.query])
