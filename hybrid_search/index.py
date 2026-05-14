"""Internal ChromaDB vector index wrapper."""

from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

from hybrid_search.chunker import Chunk

DEFAULT_COLLECTION_NAME = "hybrid_search_chunks"


def _persistent_client_class() -> Callable[..., Any]:
    from chromadb import PersistentClient

    return PersistentClient


class VectorIndex:
    """Lazily create a local persistent ChromaDB collection."""

    def __init__(
        self,
        storage_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _persistent_client_class()(path=str(self.storage_path))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        return self._collection

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks/embeddings length mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )
        if not chunks:
            return

        ids = [f"{chunk.doc_id}:{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embedding_payload = [list(vector) for vector in embeddings]
        metadatas: list[dict[str, Any]] = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "title": chunk.title,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        self.collection.add(
            ids=ids,
            embeddings=embedding_payload,
            documents=documents,
            metadatas=metadatas,
        )
