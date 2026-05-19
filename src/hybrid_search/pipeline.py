"""Internal chunk embedding pipeline helpers."""

from collections.abc import Iterable

from hybrid_search.chunker import Chunk
from hybrid_search.embedder import EmbedderLike


def embed_chunks(chunks: Iterable[Chunk], embedder: EmbedderLike) -> list[list[float]]:
    chunk_list = list(chunks)
    if not chunk_list:
        return []

    return embedder.embed_batch([chunk.text for chunk in chunk_list])
