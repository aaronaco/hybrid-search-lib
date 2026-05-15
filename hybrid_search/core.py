"""Public facade for local hybrid search."""

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from hybrid_search.bm25 import BM25Index
from hybrid_search.chunker import chunk_document
from hybrid_search.embedder import Embedder
from hybrid_search.fuzzy import FuzzyIndex
from hybrid_search.index import VectorIndex
from hybrid_search.pipeline import embed_chunks

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
    ) -> None:
        configured_path = _DEFAULT_STORAGE_PATH if storage_path is None else Path(storage_path)

        self.storage_path = configured_path.expanduser().resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.weights = dict(_DEFAULT_WEIGHTS if weights is None else weights)
        self.top_k = top_k
        self._document_ids: set[str] = set()
        self._documents: dict[str, dict[str, str]] = {}
        
        self._embedder = Embedder()
        self._vector_index = VectorIndex(storage_path=self.storage_path)
        self._bm25_index = BM25Index()
        self._fuzzy_index = FuzzyIndex()

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

    def _register_document_id(self, doc_id: str) -> None:
        self._document_ids.add(doc_id)

    def _unregister_document_id(self, doc_id: str) -> None:
        self._document_ids.discard(doc_id)

    def _has_document_id(self, doc_id: str) -> bool:
        return doc_id in self._document_ids

    def _remove_document(self, doc_id: str) -> None:
        self._unregister_document_id(doc_id)
        self._documents.pop(doc_id, None)
        self._vector_index.delete_document(doc_id)
        self._bm25_index.remove_document(doc_id)
        self._fuzzy_index.remove_document(doc_id)
