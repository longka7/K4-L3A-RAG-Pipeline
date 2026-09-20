"""Task 7 - Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Any


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse rankings by ID using the standard reciprocal-rank formula."""
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id or item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item
                first_seen[item_id] = sequence
                sequence += 1

    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))
    results = []
    for item_id in ranked_ids[:top_k]:
        result = dict(items[item_id])
        result["metadata"] = dict(items[item_id]["metadata"])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    print("Run pytest tests/test_contracts.py -q to verify RRF.")
