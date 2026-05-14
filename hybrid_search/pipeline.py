"""Internal chunk embedding pipeline helpers."""

from collections.abc import Iterable
from typing import Protocol

from hybrid_search.chunker import Chunk


class ChunkEmbedder(Protocol):
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]: ...


def embed_chunks(chunks: Iterable[Chunk], embedder: ChunkEmbedder) -> list[list[float]]:
    chunk_list = list(chunks)
    if not chunk_list:
        return []

    return embedder.embed_batch([chunk.text for chunk in chunk_list])
