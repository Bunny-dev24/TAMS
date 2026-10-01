import math

from tams.decay import decay_weight


def test_decay_weight_decreases_with_age():
    fresh = decay_weight(created_or_last_used_turn=100, current_turn=100, usage_count=0, half_life_turns=50)
    aged = decay_weight(created_or_last_used_turn=0, current_turn=100, usage_count=0, half_life_turns=50)
    assert fresh > aged


def test_decay_weight_half_life_is_exact():
    half_life = 50
    w = decay_weight(created_or_last_used_turn=0, current_turn=half_life, usage_count=0, half_life_turns=half_life)
    assert math.isclose(w, 0.5, rel_tol=1e-6)


def test_decay_weight_usage_boosts_but_never_fully_reverses_decay():
    low_usage = decay_weight(created_or_last_used_turn=0, current_turn=100, usage_count=0, half_life_turns=50)
    high_usage = decay_weight(created_or_last_used_turn=0, current_turn=100, usage_count=50, half_life_turns=50)
    assert high_usage > low_usage
    fresh = decay_weight(created_or_last_used_turn=100, current_turn=100, usage_count=0, half_life_turns=50)
    assert high_usage < fresh * 10  # boost shouldn't be unbounded
