"""Task 9 - dense, lexical, fusion and optional PageIndex fallback."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()


def _configured_threshold() -> float:
    raw_value = os.getenv("SCORE_THRESHOLD", "").strip()
    if not raw_value:
        return 0.3
    try:
        value = float(raw_value)
    except ValueError:
        return 0.3
    return min(max(value, 0.0), 1.0)


SCORE_THRESHOLD = _configured_threshold()
DEFAULT_TOP_K = 5


def calibrate_score_threshold(
    in_domain_queries: list[str],
    out_of_domain_queries: list[str],
    search_top_k: int = 5,
) -> dict[str, object]:
    """Choose the dense-score threshold with the best labeled accuracy."""
    if not in_domain_queries or not out_of_domain_queries:
        raise ValueError("calibration needs both in-domain and out-of-domain queries")
    if search_top_k <= 0:
        raise ValueError("search_top_k must be positive")

    def best_scores(queries: list[str]) -> list[float]:
        scores = []
        for query in queries:
            results = semantic_search(query, top_k=search_top_k)
            scores.append(max((float(item["score"]) for item in results), default=0.0))
        return scores

    in_scores = best_scores(in_domain_queries)
    out_scores = best_scores(out_of_domain_queries)
    candidates = {0.0, 1.0}
    ordered_scores = sorted(set(in_scores + out_scores))
    candidates.update(ordered_scores)
    candidates.update(
        (left + right) / 2.0
        for left, right in zip(ordered_scores, ordered_scores[1:])
    )

    def accuracy(threshold: float) -> float:
        correct_in = sum(score >= threshold for score in in_scores)
        correct_out = sum(score < threshold for score in out_scores)
        return (correct_in + correct_out) / (len(in_scores) + len(out_scores))

    threshold = max(candidates, key=lambda value: (accuracy(value), value))
    return {
        "threshold": threshold,
        "accuracy": accuracy(threshold),
        "in_domain_scores": in_scores,
        "out_of_domain_scores": out_scores,
    }


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieve evidence, falling back only when dense cosine confidence is low."""
    if top_k <= 0 or not query.strip():
        return []
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")

    candidate_count = max(top_k * 2, top_k)
    dense = semantic_search(query, top_k=candidate_count)
    sparse = lexical_search(query, top_k=candidate_count)
    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking
        else dense[:top_k]
    )

    best_dense_score = max((float(item["score"]) for item in dense), default=0.0)
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception:
            fallback = []
        if fallback:
            return fallback[:top_k]
    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("VinUniversity scholarship", top_k=3):
        print(result)
