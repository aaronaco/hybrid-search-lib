from collections.abc import Sequence
from pathlib import Path

import pytest

from hybrid_search import HybridSearch
from hybrid_search.chunker import Chunk
from hybrid_search.index import StoredChunk, VectorIndex


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index), float(index + 1)] for index, _ in enumerate(texts)]


def _persist_chunks(
    storage_path: Path,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    writer = VectorIndex(storage_path)
    writer.add_chunks(chunks, embeddings)
    del writer


def _stored_rows(chunks: list[StoredChunk]) -> set[tuple[str, int, str, str]]:
    return {
        (chunk.doc_id, chunk.chunk_index, chunk.title, chunk.text) for chunk in chunks
    }


def test_add_duplicate_persisted_doc_id_after_restart_raises_value_error(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    _persist_chunks(
        storage_path,
        [Chunk(doc_id="doc-1", chunk_index=0, title="Original", text="old body")],
        [[1.0, 0.0]],
    )
    search = HybridSearch(storage_path=storage_path)
    before = _stored_rows(search._vector_index.list_chunks())

    with pytest.raises(ValueError):
        search.add("doc-1", "New", "new body")

    assert _stored_rows(search._vector_index.list_chunks()) == before


def test_delete_persisted_doc_id_after_restart_removes_vector_chunks_and_unregisters(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    _persist_chunks(
        storage_path,
        [
            Chunk(doc_id="doc-1", chunk_index=0, title="Title", text="first"),
            Chunk(doc_id="doc-1", chunk_index=1, title="Title", text="second"),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    search = HybridSearch(storage_path=storage_path)

    search.delete("doc-1")

    assert search._has_document_id("doc-1") is False
    assert all(
        chunk.doc_id != "doc-1" for chunk in search._vector_index.list_chunks()
    )


def test_update_persisted_doc_id_after_restart_replaces_vector_chunks(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    _persist_chunks(
        storage_path,
        [
            Chunk(doc_id="doc-1", chunk_index=0, title="Old", text="old first"),
            Chunk(doc_id="doc-1", chunk_index=1, title="Old", text="old second"),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    search = HybridSearch(storage_path=storage_path)
    search._embedder = FakeEmbedder()  # type: ignore[assignment]

    search.update("doc-1", "New", "replacement text")

    stored_chunks = search._vector_index.list_chunks()
    assert search._has_document_id("doc-1") is True
    assert all("old" not in chunk.text for chunk in stored_chunks)
    assert any(
        chunk.doc_id == "doc-1"
        and chunk.title == "New"
        and "replacement text" in chunk.text
        for chunk in stored_chunks
    )
    assert search._bm25_index._chunk_keys == [("doc-1", 0)]
    assert search._fuzzy_index._chunk_keys == [("doc-1", 0)]


def test_unknown_doc_id_after_restart_still_raises_key_error_for_delete_and_update(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    _persist_chunks(
        storage_path,
        [Chunk(doc_id="doc-1", chunk_index=0, title="Title", text="body")],
        [[1.0, 0.0]],
    )
    search = HybridSearch(storage_path=storage_path)
    before = _stored_rows(search._vector_index.list_chunks())

    with pytest.raises(KeyError):
        search.delete("ghost")
    with pytest.raises(KeyError):
        search.update("ghost", "Ghost", "missing")

    assert search._has_document_id("doc-1") is True
    assert _stored_rows(search._vector_index.list_chunks()) == before
