from collections.abc import Sequence
from pathlib import Path

import pytest

from hybrid_search import HybridSearch, SearchResult


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = list(vector)

    def embed(self, text: str) -> list[float]:
        return list(self._vector)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def test_query_returns_persisted_semantic_result_after_restart(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    fixed_vector = [1.0, 0.0]

    writer = HybridSearch(storage_path=storage_path)
    writer._embedder = FakeEmbedder(fixed_vector)  # type: ignore[assignment]
    writer.add(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to the project",
    )
    del writer

    reader = HybridSearch(storage_path=storage_path)
    reader._embedder = FakeEmbedder(fixed_vector)  # type: ignore[assignment]

    results = reader.query("welcome")

    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)

    matched = next((r for r in results if r.doc_id == "doc-1"), None)
    assert matched is not None
    assert matched.title == "Onboarding Guide"
    assert "Welcome to the project" in matched.matched_chunk


def test_query_after_restart_has_semantic_and_bm25_contributions_only(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    fixed_vector = [1.0, 0.0]
    weights = {"semantic": 0.7, "bm25": 0.2, "fuzzy": 0.1}

    writer = HybridSearch(storage_path=storage_path, weights=weights)
    writer._embedder = FakeEmbedder(fixed_vector)  # type: ignore[assignment]
    writer.add(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to the project",
    )
    writer.add(
        doc_id="doc-2",
        title="Architecture Overview",
        content="Design notes about modules",
    )
    writer.add(
        doc_id="doc-3",
        title="Style Guide",
        content="Conventions for naming and layout",
    )
    writer.add(
        doc_id="doc-4",
        title="Release Process",
        content="How to ship a tagged version",
    )
    del writer

    reader = HybridSearch(storage_path=storage_path, weights=weights)
    reader._embedder = FakeEmbedder(fixed_vector)  # type: ignore[assignment]

    results = reader.query("welcome")

    matched = next((r for r in results if r.doc_id == "doc-1"), None)
    assert matched is not None
    assert matched.fuzzy_score == 0.0
    assert matched.semantic_score > 0.0
    assert matched.bm25_score > 0.0
    expected = (
        weights["semantic"] * matched.semantic_score
        + weights["bm25"] * matched.bm25_score
    )
    assert matched.score == pytest.approx(expected)
