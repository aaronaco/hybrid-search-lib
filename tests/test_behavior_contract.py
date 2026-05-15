from collections.abc import Sequence
from dataclasses import fields
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


def test_query_raises_value_error_when_top_k_is_zero(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path, top_k=0)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        search.query("anything")


def test_query_raises_value_error_when_top_k_is_negative(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path, top_k=-1)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        search.query("anything")


def test_query_raises_value_error_when_weights_missing_required_key(
    tmp_path: Path,
) -> None:
    search = HybridSearch(
        storage_path=tmp_path,
        weights={"semantic": 1.0},
    )
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        search.query("anything")


def test_query_raises_value_error_when_weights_has_extra_key(
    tmp_path: Path,
) -> None:
    search = HybridSearch(
        storage_path=tmp_path,
        weights={"semantic": 0.5, "bm25": 0.5, "fuzzy": 0.0, "extra": 0.1},
    )
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        search.query("anything")


def test_query_raises_value_error_when_weights_has_negative_value(
    tmp_path: Path,
) -> None:
    search = HybridSearch(
        storage_path=tmp_path,
        weights={"semantic": -0.1, "bm25": 0.5, "fuzzy": 0.6},
    )
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        search.query("anything")


def test_query_raises_value_error_when_weights_sum_to_non_positive(
    tmp_path: Path,
) -> None:
    search = HybridSearch(
        storage_path=tmp_path,
        weights={"semantic": 0.0, "bm25": 0.0, "fuzzy": 0.0},
    )
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        search.query("anything")


def test_query_with_empty_text_returns_empty_list(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    search.add(doc_id="doc-1", title="Title", content="Content")

    assert search.query("") == []


def test_query_with_whitespace_only_text_returns_empty_list(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    search.add(doc_id="doc-1", title="Title", content="Content")

    assert search.query("   ") == []


def test_add_with_duplicate_doc_id_raises_value_error(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    search.add(doc_id="doc-1", title="Title", content="Content")

    with pytest.raises(ValueError):
        search.add(doc_id="doc-1", title="Other", content="Other")


def test_delete_with_unknown_doc_id_raises_key_error(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(KeyError):
        search.delete("ghost")


def test_update_with_unknown_doc_id_raises_key_error(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(KeyError):
        search.update(doc_id="ghost", title="Title", content="Content")


def test_failed_duplicate_add_leaves_original_document_intact(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    search.add(doc_id="doc-1", title="Title", content="Content")

    with pytest.raises(ValueError):
        search.add(doc_id="doc-1", title="Other", content="Other")

    search.delete("doc-1")


def test_failed_delete_of_unknown_doc_id_does_not_corrupt_subsequent_operations(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    search.add(doc_id="doc-1", title="Onboarding Guide", content="Welcome to alpha")
    search.add(doc_id="doc-2", title="Architecture Overview", content="Design notes")
    search.add(doc_id="doc-3", title="Style Guide", content="Conventions")
    search.add(doc_id="doc-4", title="Release Process", content="Tagged versions")

    with pytest.raises(KeyError):
        search.delete("ghost")

    results = search.query("alpha")
    assert any(r.doc_id == "doc-1" for r in results)

    search.update(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to bravo",
    )
    search.delete("doc-1")


def test_add_with_persisted_doc_id_after_restart_raises_value_error(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"

    writer = HybridSearch(storage_path=storage_path)
    writer._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    writer.add(doc_id="doc-1", title="Title", content="Content")
    del writer

    reader = HybridSearch(storage_path=storage_path)
    reader._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    with pytest.raises(ValueError):
        reader.add(doc_id="doc-1", title="Other", content="Other")


def test_delete_after_restart_removes_persisted_document_from_query_results(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"

    writer = HybridSearch(storage_path=storage_path)
    writer._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    writer.add(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to the alpha project",
    )
    del writer

    reader = HybridSearch(storage_path=storage_path)
    reader._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    reader.delete("doc-1")

    results = reader.query("alpha")
    assert all(r.doc_id != "doc-1" for r in results)


def test_update_after_restart_replaces_persisted_document_content(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"

    writer = HybridSearch(storage_path=storage_path)
    writer._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    writer.add(
        doc_id="doc-1",
        title="Onboarding Guide",
        content="Welcome to the alpha project",
    )
    del writer

    reader = HybridSearch(storage_path=storage_path)
    reader._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    reader.update(
        doc_id="doc-1",
        title="New Title",
        content="new content",
    )

    results = reader.query("new")
    assert any(r.doc_id == "doc-1" for r in results)


def test_query_after_restart_returns_all_three_component_contributions(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    weights = {"semantic": 0.7, "bm25": 0.2, "fuzzy": 0.1}

    writer = HybridSearch(storage_path=storage_path, weights=weights)
    writer._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    writer.add(doc_id="doc-1", title="Onboarding Guide", content="Welcome to alpha")
    writer.add(doc_id="doc-2", title="Architecture Overview", content="Design notes")
    writer.add(doc_id="doc-3", title="Style Guide", content="Conventions")
    writer.add(doc_id="doc-4", title="Release Process", content="Tagged versions")
    del writer

    reader = HybridSearch(storage_path=storage_path, weights=weights)
    reader._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    results = reader.query("alpha")
    matched = next((r for r in results if r.doc_id == "doc-1"), None)
    assert matched is not None
    assert matched.semantic_score > 0.0
    assert matched.bm25_score > 0.0
    assert matched.fuzzy_score >= 0.0


def test_query_after_restart_with_typo_returns_fuzzy_only_keyword_contribution(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    weights = {"semantic": 0.7, "bm25": 0.2, "fuzzy": 0.1}

    writer = HybridSearch(storage_path=storage_path, weights=weights)
    writer._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    writer.add(doc_id="doc-1", title="Onboarding Guide", content="Welcome to alpha")
    writer.add(doc_id="doc-2", title="Architecture Overview", content="Design notes")
    writer.add(doc_id="doc-3", title="Style Guide", content="Conventions")
    writer.add(doc_id="doc-4", title="Release Process", content="Tagged versions")
    del writer

    reader = HybridSearch(storage_path=storage_path, weights=weights)
    reader._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    results = reader.query("alph")
    matched = next((r for r in results if r.doc_id == "doc-1"), None)
    assert matched is not None
    assert matched.fuzzy_score > 0.0
    assert matched.bm25_score == 0.0


def test_query_on_fresh_empty_index_returns_empty_list(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    assert search.query("anything") == []


def test_add_with_empty_title_and_empty_content_does_not_raise(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    search.add(doc_id="doc-1", title="", content="")


def test_add_with_empty_content_indexes_title_for_query(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]

    search.add(doc_id="doc-1", title="Some Title", content="")

    results = search.query("title")
    assert any(r.doc_id == "doc-1" for r in results)


def test_search_result_has_exactly_documented_fields(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._embedder = FakeEmbedder([1.0, 0.0])  # type: ignore[assignment]
    search.add(doc_id="doc-1", title="Onboarding Guide", content="alpha content")

    results = search.query("alpha")
    assert len(results) >= 1
    result = results[0]
    assert isinstance(result, SearchResult)

    expected = {
        "doc_id": str,
        "title": str,
        "score": float,
        "matched_chunk": str,
        "semantic_score": float,
        "bm25_score": float,
        "fuzzy_score": float,
    }
    actual = {f.name: f.type for f in fields(SearchResult)}
    assert actual == expected
