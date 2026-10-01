"""Exponential recency+usage weighting for Tier 2/3 retrieval ranking.

Decay demotes stale items from *active retrieval ranking* — it never deletes
anything. All underlying rows stay in storage (reversible, auditable).
"""
from __future__ import annotations

import math


def decay_weight(
    created_or_last_used_turn: int,
    current_turn: int,
    usage_count: int,
    half_life_turns: float,
) -> float:
    """Returns a multiplier in (0, ~1+] to apply to a similarity/BM25/rerank
    score. Recency decays exponentially with the given half-life; each prior
    usage ("this fact was retrieved and used before") slightly boosts the
    weight back up, modeling usage-based reinforcement."""
    age = max(0, current_turn - created_or_last_used_turn)
    recency = math.exp(-math.log(2) * age / max(half_life_turns, 1e-6))
    usage_boost = 1.0 + math.log1p(max(usage_count, 0)) * 0.1
    return recency * usage_boost


def apply_decay(score: float, *args, **kwargs) -> float:
    return score * decay_weight(*args, **kwargs)
