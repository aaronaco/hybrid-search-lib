from pathlib import Path

from hybrid_search import HybridSearch
from hybrid_search import embedder as embedder_module
from hybrid_search.chunker import Chunk
from hybrid_search.index import VectorIndex


def test_constructor_recovers_persisted_document_id(tmp_path: Path) -> None:
    storage_path = tmp_path / "chroma-store"
    writer = VectorIndex(storage_path)
    writer.add_chunks(
        [Chunk(doc_id="doc-1", chunk_index=0, title="Title", text="body")],
        [[1.0, 0.0]],
    )
    del writer

    search = HybridSearch(storage_path=storage_path)

    assert search._has_document_id("doc-1") is True


def test_constructor_recovers_document_id_once_for_multiple_chunks(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    writer = VectorIndex(storage_path)
    writer.add_chunks(
        [
            Chunk(doc_id="doc-1", chunk_index=0, title="Title", text="first"),
            Chunk(doc_id="doc-1", chunk_index=1, title="Title", text="second"),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    del writer

    search = HybridSearch(storage_path=storage_path)

    assert search._document_ids == {"doc-1"}


def test_constructor_recovers_empty_registry_from_empty_collection(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path / "chroma-store")

    assert search._document_ids == set()


def test_constructor_recovery_does_not_load_sentence_transformer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_if_loaded():
        raise AssertionError("sentence-transformers should not load during recovery")

    monkeypatch.setattr(
        embedder_module,
        "_sentence_transformer_class",
        fail_if_loaded,
    )

    HybridSearch(storage_path=tmp_path / "chroma-store")
