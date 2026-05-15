from importlib.metadata import metadata, packages_distributions
from pathlib import Path

from hybrid_search import HybridSearch, SearchResult
from hybrid_search import core as core_module


class EmptyVectorIndex:
    def __init__(self, storage_path):
        self.storage_path = storage_path

    def list_chunks(self):
        return []


def test_installed_metadata_exposes_public_imports() -> None:
    assert metadata("hybrid-search-lib")["Name"] == "hybrid-search-lib"
    assert "hybrid-search-lib" in packages_distributions()["hybrid_search"]
    assert HybridSearch.__name__ == "HybridSearch"
    assert SearchResult.__name__ == "SearchResult"


def test_default_construction_exposes_documented_configuration(monkeypatch) -> None:
    monkeypatch.setattr(core_module, "VectorIndex", EmptyVectorIndex)

    search = HybridSearch()

    assert search.storage_path == Path("~/.hybrid_search").expanduser().resolve()
    assert search.chunk_size == 256
    assert search.chunk_overlap == 0.15
    assert search.weights == {"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2}
    assert search.top_k == 5
