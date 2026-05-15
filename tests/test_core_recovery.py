from pathlib import Path
from typing import Any

from hybrid_search import HybridSearch
from hybrid_search import embedder as embedder_module
from hybrid_search import index as index_module
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


class _FakeCollection:
    def __init__(self, get_response: dict[str, Any]) -> None:
        self._get_response = get_response

    def get(self, *, include: list[str]) -> dict[str, Any]:
        return self._get_response


class _FakePersistentClient:
    def __init__(self, path: str, collection: _FakeCollection) -> None:
        self.path = path
        self._collection = collection

    def get_or_create_collection(self, name: str) -> _FakeCollection:
        return self._collection


def test_constructor_recovery_skips_malformed_rows_and_registers_valid_ids(
    monkeypatch, tmp_path: Path
) -> None:
    collection = _FakeCollection(
        get_response={
            "ids": ["doc-1:0", "ghost", "doc-2:0", "blank", "doc-1:1"],
            "documents": ["alpha", "ghost body", "beta", "blank body", "alpha2"],
            "metadatas": [
                {
                    "doc_id": "doc-1",
                    "chunk_index": 0,
                    "title": "A",
                    "text": "alpha",
                },
                None,
                {
                    "doc_id": "doc-2",
                    "chunk_index": 0,
                    "title": "B",
                    "text": "beta",
                },
                {"doc_id": "   ", "chunk_index": 0, "title": "X", "text": "blank"},
                {
                    "doc_id": "doc-1",
                    "chunk_index": 1,
                    "title": "A",
                    "text": "alpha2",
                },
            ],
        }
    )
    monkeypatch.setattr(
        index_module,
        "_persistent_client_class",
        lambda: lambda path: _FakePersistentClient(path, collection),
    )

    search = HybridSearch(storage_path=tmp_path / "chroma-store")

    assert search._document_ids == {"doc-1", "doc-2"}
