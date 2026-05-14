from pathlib import Path
from typing import Any

import pytest

from hybrid_search import index as index_module
from hybrid_search.chunker import Chunk
from hybrid_search.index import DEFAULT_COLLECTION_NAME, VectorIndex


class FakeCollection:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.add_calls.append(
            {
                "ids": list(ids),
                "embeddings": [list(vec) for vec in embeddings],
                "documents": list(documents),
                "metadatas": list(metadatas),
            }
        )


class FakePersistentClient:
    instances: list["FakePersistentClient"] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self.collection_names: list[str] = []
        self.collection = FakeCollection()
        self.instances.append(self)

    def get_or_create_collection(self, name: str) -> FakeCollection:
        self.collection_names.append(name)
        return self.collection


def use_fake_persistent_client(monkeypatch) -> None:
    FakePersistentClient.instances = []
    monkeypatch.setattr(
        index_module,
        "_persistent_client_class",
        lambda: FakePersistentClient,
    )


def test_vector_index_construction_does_not_initialize_client(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)

    VectorIndex(tmp_path)

    assert FakePersistentClient.instances == []


def test_vector_index_collection_initializes_one_client(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)
    index = VectorIndex(tmp_path)

    collection = index.collection

    assert len(FakePersistentClient.instances) == 1
    assert collection is FakePersistentClient.instances[0].collection


def test_vector_index_uses_default_collection_name(tmp_path: Path) -> None:
    index = VectorIndex(tmp_path)

    assert index.collection_name == DEFAULT_COLLECTION_NAME
    assert index.collection_name == "hybrid_search_chunks"


def test_vector_index_resolves_storage_path(tmp_path: Path) -> None:
    storage_path = tmp_path / "nested" / ".." / "index"

    index = VectorIndex(storage_path)

    assert index.storage_path == storage_path.expanduser().resolve()


def test_vector_index_collection_uses_configured_collection_name(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)
    index = VectorIndex(tmp_path, collection_name="custom_chunks")

    index.collection

    assert FakePersistentClient.instances[0].collection_names == ["custom_chunks"]


def test_add_chunks_stores_one_entry_per_chunk_with_deterministic_ids(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)
    index = VectorIndex(tmp_path)
    chunks = [
        Chunk(doc_id="doc-1", chunk_index=0, title="Title", text="first chunk"),
        Chunk(doc_id="doc-1", chunk_index=1, title="Title", text="second chunk"),
    ]

    index.add_chunks(chunks, [[0.0, 1.0], [2.0, 3.0]])

    collection = FakePersistentClient.instances[0].collection
    assert len(collection.add_calls) == 1
    call = collection.add_calls[0]
    assert call["ids"] == ["doc-1:0", "doc-1:1"]
    assert call["documents"] == ["first chunk", "second chunk"]
    assert call["embeddings"] == [[0.0, 1.0], [2.0, 3.0]]


def test_add_chunks_records_required_metadata_per_chunk(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)
    index = VectorIndex(tmp_path)
    chunk = Chunk(doc_id="doc-7", chunk_index=3, title="Heading", text="body text")

    index.add_chunks([chunk], [[0.5, 0.25]])

    collection = FakePersistentClient.instances[0].collection
    assert collection.add_calls[0]["metadatas"] == [
        {
            "doc_id": "doc-7",
            "chunk_index": 3,
            "title": "Heading",
            "text": "body text",
        }
    ]


def test_add_chunks_raises_when_chunk_and_embedding_counts_differ(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)
    index = VectorIndex(tmp_path)
    chunks = [
        Chunk(doc_id="doc-1", chunk_index=0, title="t", text="a"),
        Chunk(doc_id="doc-1", chunk_index=1, title="t", text="b"),
    ]

    with pytest.raises(ValueError):
        index.add_chunks(chunks, [[0.0]])

    with pytest.raises(ValueError):
        index.add_chunks(chunks[:1], [[0.0], [1.0]])

    assert FakePersistentClient.instances == []


def test_add_chunks_with_empty_input_does_not_initialize_client(
    monkeypatch, tmp_path: Path
) -> None:
    use_fake_persistent_client(monkeypatch)
    index = VectorIndex(tmp_path)

    index.add_chunks([], [])

    assert FakePersistentClient.instances == []
