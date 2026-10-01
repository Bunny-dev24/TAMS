from tams.storage.fact_db import FactStore


def test_write_and_read_numeric_fact():
    fs = FactStore(session_id="s1")
    fs.write("numeric", turn_id=1, confidence=0.9, param="timeout", value="30", unit="s")
    facts = fs.active_facts("numeric")
    assert len(facts) == 1
    assert facts[0]["param"] == "timeout"
    assert facts[0]["value"] == "30"


def test_conflict_resolution_keeps_both_marks_old_superseded():
    fs = FactStore(session_id="s1")
    r1 = fs.write("numeric", turn_id=1, confidence=0.9, param="timeout", value="30", unit="s")
    r2 = fs.write("numeric", turn_id=5, confidence=0.95, param="timeout", value="60", unit="s")

    assert r2.conflicted_with == r1.fact_id
    active = fs.active_facts("numeric")
    assert len(active) == 1
    assert active[0]["value"] == "60"

    # Old fact still exists in storage (never silently deleted) — exported.
    export = fs.export_all()
    all_numeric_ids = {f["id"] for f in export["numeric"]}
    assert r1.fact_id in all_numeric_ids
    assert r2.fact_id in all_numeric_ids
    old = next(f for f in export["numeric"] if f["id"] == r1.fact_id)
    assert old["status"] == "superseded"
    assert old["superseded_by"] == r2.fact_id


def test_match_entities_deterministic_lookup():
    fs = FactStore(session_id="s1")
    fs.write("dependency", turn_id=1, confidence=0.8, entity="AuthModule", modified_at_turn=1, referenced_at_turns=[])
    hits = fs.match_entities("dependency", ["AuthModule"])
    assert len(hits) == 1
    assert hits[0]["entity"] == "AuthModule"


def test_dependency_reference_appends_without_overwriting_modification():
    fs = FactStore(session_id="s1")
    r1 = fs.write("dependency", turn_id=2, confidence=0.8, entity="PaymentGateway", modified_at_turn=2, referenced_at_turns=[])
    fs.record_dependency_reference("PaymentGateway", 7)
    fs.record_dependency_reference("PaymentGateway", 9)
    active = fs.active_facts("dependency")
    assert len(active) == 1
    assert active[0]["modified_at_turn"] == 2
    assert active[0]["referenced_at_turns"] == [7, 9]


def test_hitl_correction_marks_old_superseded_and_logs():
    fs = FactStore(session_id="s1")
    r1 = fs.write("numeric", turn_id=1, confidence=0.5, param="budget", value="100", unit="usd")
    new_id = fs.correct("numeric", r1.fact_id, {"value": "150"}, note="user corrected typo")

    active = fs.active_facts("numeric")
    assert len(active) == 1
    assert active[0]["value"] == "150"
    assert active[0]["id"] == new_id

    export = fs.export_all()
    old = next(f for f in export["numeric"] if f["id"] == r1.fact_id)
    assert old["status"] == "superseded"
    assert old["superseded_by"] == new_id


def test_privacy_export_and_delete():
    fs = FactStore(session_id="s1")
    fs.write("numeric", turn_id=1, confidence=0.9, param="x", value="1", unit=None)
    fs.write("constraint", turn_id=1, confidence=0.9, constraint_text="must not do X", scope="global", reason=None)

    export = fs.export_all()
    assert len(export["numeric"]) == 1
    assert len(export["constraint"]) == 1

    deleted = fs.delete_all()
    assert deleted == 2
    assert fs.active_facts("numeric") == []
    assert fs.active_facts("constraint") == []
