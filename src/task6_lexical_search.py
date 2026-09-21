"""Task 6 - lexical retrieval over the same chunks used by dense search."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Plus

from .contracts import validate_document
from .task4_chunking_indexing import get_collection


CORPUS: list[dict] = []
_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def _corpus_from_vectorstore() -> list[dict]:
    response = get_collection().get(include=["documents", "metadatas"])
    ids = response.get("ids") or []
    documents = response.get("documents") or []
    metadatas = response.get("metadatas") or []
    corpus = []
    for item_id, content, metadata in zip(ids, documents, metadatas):
        metadata = {**(metadata or {}), "url": (metadata or {}).get("url")}
        item = {"id": item_id, "content": content, "metadata": metadata}
        validate_document(item, require_chunk=True)
        corpus.append(item)
    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Plus:
    """Build a BM25 index from valid, non-empty chunks."""
    if not corpus:
        raise ValueError("cannot build a BM25 index from an empty corpus")
    for item in corpus:
        validate_document(item, require_chunk=True)
    return BM25Plus([_tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique BM25 results sorted by descending score."""
    if top_k <= 0 or not query.strip():
        return []

    corpus = CORPUS if CORPUS else _corpus_from_vectorstore()
    if not corpus:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = build_bm25_index(corpus).get_scores(query_tokens)
    query_token_set = set(query_tokens)
    matching_indices = [
        index
        for index, item in enumerate(corpus)
        if query_token_set.intersection(_tokenize(item["content"]))
    ]
    ranked_indices = sorted(
        matching_indices, key=lambda index: (-float(scores[index]), index)
    )
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index in ranked_indices:
        score = float(scores[index])
        item = corpus[index]
        if score <= 0 or item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": dict(item["metadata"]),
            "retrieval_method": "bm25",
        })
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("scholarship financial aid", top_k=3):
        print(result)
