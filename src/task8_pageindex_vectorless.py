"""Task 8 - optional PageIndex fallback with a persistent document-ID cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_API_URL = os.getenv("PAGEINDEX_API_URL", "https://api.pageindex.ai").rstrip("/")
REQUEST_TIMEOUT = 30
ROOT_DIR = Path(__file__).parent.parent
LANDING_LEGAL_DIR = ROOT_DIR / "data" / "landing" / "legal"
CACHE_PATH = ROOT_DIR / "data" / ".pageindex_documents.json"
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".pptx"}


def _load_cache() -> dict[str, dict[str, str]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: dict[str, dict[str, str]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(CACHE_PATH)


def _source_files() -> list[Path]:
    if not LANDING_LEGAL_DIR.exists():
        return []
    return sorted(
        path
        for path in LANDING_LEGAL_DIR.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )


def upload_documents() -> None:
    """Upload new policy files and cache PageIndex IDs by resolved source path."""
    if not PAGEINDEX_API_KEY:
        return

    cache = _load_cache()
    changed = False
    for path in _source_files():
        source_key = str(path.resolve())
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        cached = cache.get(source_key, {})
        if cached.get("sha256") == fingerprint and cached.get("doc_id"):
            continue

        with path.open("rb") as file_handle:
            response = requests.post(
                f"{PAGEINDEX_API_URL}/doc/",
                headers={"api_key": PAGEINDEX_API_KEY},
                files={"file": (path.name, file_handle)},
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        payload = response.json()
        doc_id = payload.get("doc_id") or payload.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {path.name}")
        cache[source_key] = {
            "doc_id": doc_id,
            "source": path.name,
            "sha256": fingerprint,
        }
        changed = True

    if changed:
        _save_cache(cache)


def _citation_content(citation: dict[str, Any], answer: str) -> str:
    for key in ("text", "content", "quote", "excerpt"):
        value = citation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return answer.strip()


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query indexed documents and map PageIndex citations to SearchResult."""
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY:
        return []

    upload_documents()
    cache = _load_cache()
    entries = list(cache.values())
    doc_ids = [entry["doc_id"] for entry in entries if entry.get("doc_id")]
    if not doc_ids:
        return []

    response = requests.post(
        f"{PAGEINDEX_API_URL}/chat/completions",
        headers={
            "api_key": PAGEINDEX_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "messages": [{"role": "user", "content": query}],
            "doc_id": doc_ids,
            "stream": False,
            "enable_citations": True,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    answer = message.get("content", "") if isinstance(message, dict) else ""
    citations = payload.get("citations") or message.get("citations") or []
    source_by_id = {entry.get("doc_id"): entry.get("source") for entry in entries}

    results = []
    seen_ids: set[str] = set()
    for rank, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            continue
        doc_id = citation.get("doc_id") or citation.get("document_id")
        source = (
            citation.get("source")
            or citation.get("file")
            or citation.get("filename")
            or source_by_id.get(doc_id)
            or str(doc_id or "pageindex")
        )
        page = citation.get("page") or citation.get("page_number")
        block = citation.get("block") or citation.get("block_id") or rank
        result_id = f"pageindex:{doc_id or source}:{page or 0}:{block}"
        if result_id in seen_ids:
            continue
        seen_ids.add(result_id)
        results.append({
            "id": result_id,
            "content": _citation_content(citation, str(answer)),
            "score": 1.0 / rank,
            "metadata": {
                "source": str(source),
                "title": str(citation.get("title") or Path(str(source)).stem),
                "doc_type": "legal",
                "url": citation.get("url"),
                "chunk_index": rank - 1,
            },
            "retrieval_method": "pageindex",
        })
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    upload_documents()
