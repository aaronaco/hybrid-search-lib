from hybrid_search.chunker import Chunk
from hybrid_search.core import _stored_chunk_to_chunk, _stored_chunks_to_chunks
from hybrid_search.index import StoredChunk


def test_stored_chunk_to_chunk_preserves_all_fields() -> None:
    stored = StoredChunk(
        doc_id="doc-1",
        chunk_index=3,
        title="Onboarding Guide",
        text="Welcome to the project",
    )

    result = _stored_chunk_to_chunk(stored)

    assert result == Chunk(
        doc_id="doc-1",
        chunk_index=3,
        title="Onboarding Guide",
        text="Welcome to the project",
    )


def test_stored_chunks_to_chunks_preserves_order() -> None:
    stored = [
        StoredChunk(doc_id="doc-1", chunk_index=0, title="A", text="alpha"),
        StoredChunk(doc_id="doc-2", chunk_index=1, title="B", text="beta"),
        StoredChunk(doc_id="doc-3", chunk_index=2, title="C", text="gamma"),
    ]

    result = _stored_chunks_to_chunks(stored)

    assert result == [
        Chunk(doc_id="doc-1", chunk_index=0, title="A", text="alpha"),
        Chunk(doc_id="doc-2", chunk_index=1, title="B", text="beta"),
        Chunk(doc_id="doc-3", chunk_index=2, title="C", text="gamma"),
    ]


def test_stored_chunks_to_chunks_returns_empty_list_for_empty_iterable() -> None:
    assert _stored_chunks_to_chunks([]) == []
