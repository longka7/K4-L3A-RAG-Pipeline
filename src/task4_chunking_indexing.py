"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import re
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    class RecursiveCharacterTextSplitter:
        """Fallback local splitter when langchain_text_splitters is not installed."""

        def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators

        def split_text(self, text: str) -> list[str]:
            cleaned = text.strip()
            if not cleaned:
                return []
            if len(cleaned) <= self.chunk_size:
                return [cleaned]

            chunks: list[str] = []
            start = 0
            while start < len(cleaned):
                end = min(len(cleaned), start + self.chunk_size)
                if end < len(cleaned):
                    best_index = -1
                    for sep in self.separators:
                        if sep in ("", " "):
                            continue
                        index = cleaned.rfind(sep, start, end)
                        if index > start and index > best_index:
                            best_index = index
                    if best_index > start:
                        end = best_index + 1
                chunk = cleaned[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(cleaned):
                    break
                start = max(start + self.chunk_size - self.chunk_overlap, end - self.chunk_overlap)
                if start <= 0:
                    start = end
            return chunks


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


_embedding_model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng cùng model dùng cho dense retrieval."""
    global _embedding_model
    if not texts:
        return []

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = _embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    # TODO: Tạo hoặc mở persistent collection.
    #
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # raise NotImplementedError("Implement get_collection")


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    # TODO: Đọc mọi .md và tạo Document theo contract.
    #
    documents = []
    for path in STANDARDIZED_DIR.rglob("*.md"):
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": path.read_text(encoding="utf-8"),
            "metadata": {
                "source": path.name,
                "title": path.stem,
                "doc_type": doc_type,
                "url": None,
            },
        })
    return documents
    # raise NotImplementedError("Implement load_documents")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            chunk_text = text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": chunk_text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    # TODO: Embed theo batch và giữ nguyên các field của chunk.
    #
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks
    # raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    # TODO: Upsert ids, documents, embeddings và metadatas.
    #
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    # raise NotImplementedError("Implement index_to_vectorstore")


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
