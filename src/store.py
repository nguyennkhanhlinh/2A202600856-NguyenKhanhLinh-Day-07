from __future__ import annotations

import math
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity, robust to non-normalized vectors. Returns 0.0 for zero vectors."""
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.

    The in-memory list ``self._store`` is always the source of truth for
    search/filter/delete so behavior stays deterministic; ChromaDB is mirrored
    for optional persistence when it is installed.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        metadata = dict(doc.metadata or {})
        metadata["doc_id"] = doc.id
        record_id = f"{doc.id}::{self._next_index}"
        self._next_index += 1
        return {
            "id": record_id,
            "doc_id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": metadata,
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Run in-memory similarity search over the provided records."""
        query_embedding = self._embedding_fn(query)
        scored = [
            {
                "content": record["content"],
                "score": _cosine(query_embedding, record["embedding"]),
                "metadata": record["metadata"],
            }
            for record in records
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it."""
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

            if self._use_chroma and self._collection is not None:
                try:
                    self._collection.add(
                        ids=[record["id"]],
                        documents=[record["content"]],
                        embeddings=[record["embedding"]],
                        metadatas=[record["metadata"]],
                    )
                except Exception:
                    # Persistence is best-effort; the in-memory store still holds the data.
                    pass

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict = None
    ) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if metadata_filter:
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        else:
            candidates = self._store

        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document. Returns True if any were removed."""
        remaining = [record for record in self._store if record["doc_id"] != doc_id]
        removed = len(remaining) != len(self._store)

        if removed:
            ids_to_delete = [
                record["id"] for record in self._store if record["doc_id"] == doc_id
            ]
            self._store = remaining

            if self._use_chroma and self._collection is not None:
                try:
                    self._collection.delete(ids=ids_to_delete)
                except Exception:
                    pass

        return removed
