from collections.abc import Iterable

from hybrid_search.chunker import Chunk
from hybrid_search.pipeline import embed_chunks


class FakeEmbedder:
    def __init__(self) -> None:
        self.received_texts: list[list[str]] = []

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        self.received_texts.append(text_list)
        return [[float(index)] for index, _ in enumerate(text_list)]


def test_embed_chunks_returns_one_vector_per_chunk() -> None:
    embedder = FakeEmbedder()
    chunks = [
        Chunk(doc_id="doc-1", chunk_index=0, title="t", text="first chunk"),
        Chunk(doc_id="doc-1", chunk_index=1, title="t", text="second chunk"),
    ]

    vectors = embed_chunks(chunks, embedder)

    assert vectors == [[0.0], [1.0]]
    assert embedder.received_texts == [["first chunk", "second chunk"]]


def test_embed_chunks_returns_empty_list_without_calling_embedder() -> None:
    embedder = FakeEmbedder()

    vectors = embed_chunks([], embedder)

    assert vectors == []
    assert embedder.received_texts == []


def test_embed_chunks_preserves_chunk_order() -> None:
    embedder = FakeEmbedder()
    chunks = [
        Chunk(doc_id="doc-1", chunk_index=0, title="t", text="z last lexically"),
        Chunk(doc_id="doc-1", chunk_index=1, title="t", text="a first lexically"),
        Chunk(doc_id="doc-1", chunk_index=2, title="t", text="m middle lexically"),
    ]

    vectors = embed_chunks(chunks, embedder)

    assert vectors == [[0.0], [1.0], [2.0]]
    assert embedder.received_texts == [
        ["z last lexically", "a first lexically", "m middle lexically"]
    ]
