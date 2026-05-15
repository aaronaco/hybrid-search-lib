from pathlib import Path

from hybrid_search import HybridSearch
from hybrid_search import core as core_module


class EmptyVectorIndex:
    def __init__(self, storage_path):
        self.storage_path = storage_path

    def list_chunks(self):
        return []


def use_empty_vector_index(monkeypatch) -> None:
    monkeypatch.setattr(core_module, "VectorIndex", EmptyVectorIndex)


def test_constructor_uses_documented_defaults(monkeypatch) -> None:
    use_empty_vector_index(monkeypatch)

    search = HybridSearch()

    assert search.storage_path == Path("~/.hybrid_search").expanduser().resolve()
    assert search.chunk_size == 256
    assert search.chunk_overlap == 0.15
    assert search.weights == {"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2}
    assert search.top_k == 5


def test_constructor_normalizes_explicit_storage_path(tmp_path: Path) -> None:
    storage_path = tmp_path / "index" / ".." / "index"

    search = HybridSearch(storage_path=storage_path)

    assert search.storage_path == storage_path.expanduser().resolve()


def test_default_weights_are_not_shared_between_instances(monkeypatch) -> None:
    use_empty_vector_index(monkeypatch)

    first = HybridSearch()
    second = HybridSearch()

    first.weights["semantic"] = 1.0

    assert second.weights == {"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2}
