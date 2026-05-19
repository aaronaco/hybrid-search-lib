"""Public facade for local hybrid search."""

from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from hybrid_search.bm25 import BM25Index
from hybrid_search.chunker import Chunk, chunk_document
from hybrid_search.embedder import DEFAULT_EMBEDDING_MODEL, Embedder, EmbedderLike
from hybrid_search.fuzzy import FuzzyIndex
from hybrid_search.index import StoredChunk, VectorIndex
from hybrid_search.pipeline import embed_chunks
from hybrid_search.ranker import rank
from hybrid_search.result import SearchResult

_DEFAULT_STORAGE_PATH = Path("~/.hybrid_search")
_DEFAULT_WEIGHTS = MappingProxyType({"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2})


class HybridSearch:
    """Configure a local hybrid search index."""

    def __init__(
        self,
        storage_path: str | Path | None = None,
        chunk_size: int = 256,
        chunk_overlap: float = 0.15,
        weights: Mapping[str, float] | None = None,
        top_k: int = 5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedder: EmbedderLike | None = None,
    ) -> None:
        configured_path = _DEFAULT_STORAGE_PATH if storage_path is None else Path(storage_path)

        self.storage_path = configured_path.expanduser().resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.weights = dict(_DEFAULT_WEIGHTS if weights is None else weights)
        self.top_k = top_k
        self._document_ids: set[str] = set()
        self._documents: dict[str, dict[str, str]] = {}
        
        self._embedder = (
            Embedder(model_name=embedding_model) if embedder is None else embedder
        )
        self._vector_index = VectorIndex(storage_path=self.storage_path)
        self._bm25_index = BM25Index()
        self._fuzzy_index = FuzzyIndex()
        self._recover_from_persistence()

    def add(self, doc_id: str, title: str, content: str) -> None:
        if self._has_document_id(doc_id):
            raise ValueError(f"Document already exists: {doc_id}")
        self._register_document_id(doc_id)
        self._documents[doc_id] = {"title": title, "content": content}
        
        chunks = chunk_document(
            doc_id=doc_id,
            title=title,
            content=content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        embeddings = embed_chunks(chunks, self._embedder)
        self._vector_index.add_chunks(chunks, embeddings)
        self._bm25_index.add_chunks(chunks)
        self._fuzzy_index.add_chunks(chunks)

    def update(self, doc_id: str, title: str, content: str) -> None:
        if not self._has_document_id(doc_id):
            raise KeyError(f"Document not found: {doc_id}")
        self._remove_document(doc_id)
        self.add(doc_id, title, content)

    def delete(self, doc_id: str) -> None:
        if not self._has_document_id(doc_id):
            raise KeyError(f"Document not found: {doc_id}")
        self._remove_document(doc_id)

    def query(self, text: str) -> list[SearchResult]:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive: got {self.top_k}")
        weight_keys = set(self.weights.keys())
        if weight_keys != {"semantic", "bm25", "fuzzy"}:
            raise ValueError(
                "weights must contain exactly keys 'semantic', 'bm25', 'fuzzy': "
                f"got {sorted(weight_keys)}"
            )
        if any(v < 0 for v in self.weights.values()):
            raise ValueError(
                f"weights values must be non-negative: got {dict(self.weights)}"
            )
        if sum(self.weights.values()) <= 0:
            raise ValueError(
                f"weights must sum to a positive number: got {dict(self.weights)}"
            )

        if not text.strip():
            return []

        candidate_size = max(self.top_k * 4, 20)
        query_vector = self._embedder.embed(text)
        semantic_matches = self._vector_index.query(query_vector, top_k=candidate_size)
        bm25_matches = self._bm25_index.search(text, top_k=candidate_size)
        fuzzy_matches = self._fuzzy_index.search(text, top_k=candidate_size)

        return rank(
            semantic_matches,
            bm25_matches,
            fuzzy_matches,
            self.weights,
            self.top_k,
        )

    def _register_document_id(self, doc_id: str) -> None:
        self._document_ids.add(doc_id)

    def _unregister_document_id(self, doc_id: str) -> None:
        self._document_ids.discard(doc_id)

    def _has_document_id(self, doc_id: str) -> bool:
        return doc_id in self._document_ids

    def _recover_from_persistence(self) -> None:
        stored = list(self._vector_index.list_chunks())
        for item in stored:
            self._register_document_id(item.doc_id)
        chunks = _stored_chunks_to_chunks(stored)
        self._bm25_index.add_chunks(chunks)
        self._fuzzy_index.add_chunks(chunks)

    def _remove_document(self, doc_id: str) -> None:
        self._unregister_document_id(doc_id)
        self._documents.pop(doc_id, None)
        self._vector_index.delete_document(doc_id)
        self._bm25_index.remove_document(doc_id)
        self._fuzzy_index.remove_document(doc_id)


def _stored_chunk_to_chunk(stored: StoredChunk) -> Chunk:
    return Chunk(
        doc_id=stored.doc_id,
        chunk_index=stored.chunk_index,
        title=stored.title,
        text=stored.text,
    )


def _stored_chunks_to_chunks(stored: Iterable[StoredChunk]) -> list[Chunk]:
    return [_stored_chunk_to_chunk(item) for item in stored]
