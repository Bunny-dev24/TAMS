from tams.storage.working_memory import InMemoryWorkingMemoryStore, Turn
from tams.tiers.tier1_working_memory import WorkingMemory


def test_append_and_get_all(settings):
    wm = WorkingMemory(InMemoryWorkingMemoryStore(), settings)
    wm.append(Turn(turn_id=1, session_id="s1", speaker="user", content="hello"))
    wm.append(Turn(turn_id=2, session_id="s1", speaker="assistant", content="hi there"))
    turns = wm.get_all("s1")
    assert len(turns) == 2
    assert turns[0].content == "hello"


def test_needs_compaction_on_turn_count(settings):
    wm = WorkingMemory(InMemoryWorkingMemoryStore(), settings)
    for i in range(settings.working_memory_max_turns + 1):
        wm.append(Turn(turn_id=i, session_id="s1", speaker="user", content="short"))
    assert wm.needs_compaction("s1") is True


def test_evict_for_compaction_removes_oldest_first(settings):
    wm = WorkingMemory(InMemoryWorkingMemoryStore(), settings)
    for i in range(10):
        wm.append(Turn(turn_id=i, session_id="s1", speaker="user", content="x" * 50))
    evicted = wm.evict_for_compaction("s1", target_fraction=0.0)
    assert len(evicted) > 0
    assert evicted[0].turn_id == 0  # oldest evicted first
    remaining = wm.get_all("s1")
    remaining_ids = {t.turn_id for t in remaining}
    evicted_ids = {t.turn_id for t in evicted}
    assert remaining_ids.isdisjoint(evicted_ids)
