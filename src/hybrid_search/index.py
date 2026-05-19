"""Internal ChromaDB vector index wrapper."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

from hybrid_search.chunker import Chunk

DEFAULT_COLLECTION_NAME = "hybrid_search_chunks"


@dataclass
class SemanticMatch:
    """Internal semantic query result carrying chunk identity and raw distance."""

    doc_id: str
    chunk_index: int
    title: str
    text: str
    distance: float


@dataclass
class StoredChunk:
    """Internal stored chunk metadata recovered from the vector index."""

    doc_id: str
    chunk_index: int
    title: str
    text: str


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

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
    ) -> list[SemanticMatch]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive: got {top_k}")

        result = self.collection.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
        )

        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        matches: list[SemanticMatch] = []
        for metadata, distance in zip(metadatas, distances):
            matches.append(
                SemanticMatch(
                    doc_id=metadata["doc_id"],
                    chunk_index=metadata["chunk_index"],
                    title=metadata["title"],
                    text=metadata["text"],
                    distance=float(distance),
                )
            )
        return matches

    def list_chunks(self) -> list[StoredChunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        metadatas = result.get("metadatas") or []

        chunks: list[StoredChunk] = []
        for metadata in metadatas:
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id.strip():
                continue
            chunks.append(
                StoredChunk(
                    doc_id=doc_id,
                    chunk_index=int(metadata["chunk_index"]),
                    title=str(metadata.get("title", "")),
                    text=str(metadata.get("text", "")),
                )
            )
        return chunks

    def delete_document(self, doc_id: str) -> None:
        if not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        self.collection.delete(where={"doc_id": doc_id})
