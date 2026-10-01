from tams.eval.harness import run_full_context_baseline, run_tams_eval
from tams.eval.synth_data import generate_haystack


def test_needle_haystack_generation_has_queries_per_category():
    haystack = generate_haystack(num_filler_turns=60, seed=1)
    categories = {n.category for n in haystack.needles}
    assert "numeric" in categories
    assert "constraint" in categories
    assert "decision" in categories
    assert "dependency" in categories


def test_full_context_baseline_has_perfect_recall():
    haystack = generate_haystack(num_filler_turns=60, seed=2)
    report = run_full_context_baseline(haystack, settings=None)
    assert report.overall_recall_at_k == 1.0


def test_tams_eval_runs_and_reports_recall_per_category(settings):
    haystack = generate_haystack(num_filler_turns=40, seed=3)
    report = run_tams_eval(haystack, settings)
    assert 0.0 <= report.overall_recall_at_k <= 1.0
    assert report.compaction_token_reduction_ratio is not None
    for cat, rec in report.recall_by_category.items():
        assert 0.0 <= rec.recall_at_k <= 1.0


def test_ablation_no_extractors_reduces_structured_recall(settings):
    haystack = generate_haystack(num_filler_turns=60, seed=4)
    with_extractors = run_tams_eval(haystack, settings, enable_extractors=True, session_id="abl-with")
    without_extractors = run_tams_eval(haystack, settings, enable_extractors=False, session_id="abl-without")
    # Without dedicated extractors, deterministic hard-fact hits for numeric/
    # constraint/decision/dependency should not exceed the with-extractors run.
    assert without_extractors.overall_recall_at_k <= with_extractors.overall_recall_at_k + 1e-9
