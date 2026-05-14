from pathlib import Path

from hybrid_search import index as index_module
from hybrid_search.index import DEFAULT_COLLECTION_NAME, VectorIndex


class FakeCollection:
    pass


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
