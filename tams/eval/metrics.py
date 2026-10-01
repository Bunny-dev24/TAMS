"""Eval metrics: retrieval precision@k / recall@k (per category), citation
accuracy, hallucination rate, compaction token-reduction ratio, latency."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CategoryRecall:
    category: str
    hits: int = 0
    total: int = 0

    @property
    def recall_at_k(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class EvalReport:
    recall_by_category: dict[str, CategoryRecall] = field(default_factory=dict)
    overall_recall_at_k: float = 0.0
    avg_latency_s: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_citation_coverage: float = 0.0
    compaction_token_reduction_ratio: float | None = None

    def to_dict(self) -> dict:
        return {
            "recall_by_category": {
                c: {"recall_at_k": r.recall_at_k, "hits": r.hits, "total": r.total}
                for c, r in self.recall_by_category.items()
            },
            "overall_recall_at_k": self.overall_recall_at_k,
            "avg_latency_s": self.avg_latency_s,
            "avg_hallucination_rate": self.avg_hallucination_rate,
            "avg_citation_coverage": self.avg_citation_coverage,
            "compaction_token_reduction_ratio": self.compaction_token_reduction_ratio,
        }


def check_hit(expected_fragment: str, context_text: str) -> bool:
    return expected_fragment.lower() in context_text.lower()


def aggregate_results(
    per_query_results: list[dict],
) -> EvalReport:
    """per_query_results: [{category, hit: bool, latency_s, hallucination_rate, citation_coverage}]"""
    by_cat: dict[str, CategoryRecall] = defaultdict(lambda: CategoryRecall(category=""))
    latencies = []
    hallucinations = []
    citations = []
    total_hits = 0

    for r in per_query_results:
        cat = r["category"]
        bucket = by_cat[cat]
        bucket.category = cat
        bucket.total += 1
        if r["hit"]:
            bucket.hits += 1
            total_hits += 1
        latencies.append(r.get("latency_s", 0.0))
        if "hallucination_rate" in r:
            hallucinations.append(r["hallucination_rate"])
        if "citation_coverage" in r:
            citations.append(r["citation_coverage"])

    n = len(per_query_results) or 1
    return EvalReport(
        recall_by_category=dict(by_cat),
        overall_recall_at_k=total_hits / n,
        avg_latency_s=sum(latencies) / n,
        avg_hallucination_rate=(sum(hallucinations) / len(hallucinations)) if hallucinations else 0.0,
        avg_citation_coverage=(sum(citations) / len(citations)) if citations else 0.0,
    )


def compaction_ratio(raw_tokens: int, compacted_tokens: int) -> float:
    if raw_tokens == 0:
        return 0.0
    return 1.0 - (compacted_tokens / raw_tokens)
