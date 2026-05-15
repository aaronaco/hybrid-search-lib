from collections.abc import Sequence
from pathlib import Path

from hybrid_search import HybridSearch, SearchResult


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = list(vector)

    def embed(self, text: str) -> list[float]:
        return list(self._vector)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def test_lifecycle_add_query_update_delete_round_trip_through_public_api(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    search.add(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to the alpha project",
    )
    search.add(
        doc_id="doc-2",
        title="Architecture Overview",
        content="Design notes about modules",
    )
    search.add(
        doc_id="doc-3",
        title="Style Guide",
        content="Conventions for naming",
    )
    search.add(
        doc_id="doc-4",
        title="Release Process",
        content="How to ship a tagged version",
    )

    results = search.query("alpha")
    assert any(r.doc_id == "doc-1" for r in results)
    assert all(isinstance(r, SearchResult) for r in results)

    search.update(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to the bravo project",
    )

    results_after_update = search.query("bravo")
    assert any(r.doc_id == "doc-1" for r in results_after_update)

    search.delete("doc-1")

    results_after_delete = search.query("bravo")
    assert all(r.doc_id != "doc-1" for r in results_after_delete)
