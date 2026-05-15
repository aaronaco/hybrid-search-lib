import pytest

from hybrid_search.bm25 import BM25Match
from hybrid_search.fuzzy import FuzzyMatch
from hybrid_search.index import SemanticMatch
from hybrid_search.ranker import (
    normalize_bm25,
    normalize_fuzzy,
    normalize_semantic,
)


def _semantic(distance: float, doc_id: str = "d", chunk_index: int = 0) -> SemanticMatch:
    return SemanticMatch(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title="t",
        text="x",
        distance=distance,
    )


def _bm25(score: float, doc_id: str = "d", chunk_index: int = 0) -> BM25Match:
    return BM25Match(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title="t",
        text="x",
        score=score,
    )


def _fuzzy(score: float, doc_id: str = "d", chunk_index: int = 0) -> FuzzyMatch:
    return FuzzyMatch(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title="t",
        text="x",
        score=score,
    )


def test_normalize_semantic_maps_distance_to_score() -> None:
    matches = [
        _semantic(distance=0.0, doc_id="a", chunk_index=0),
        _semantic(distance=0.5, doc_id="b", chunk_index=1),
        _semantic(distance=2.0, doc_id="c", chunk_index=2),
    ]

    result = normalize_semantic(matches)

    assert result == [
        (("a", 0), pytest.approx(1.0)),
        (("b", 1), pytest.approx(0.75)),
        (("c", 2), pytest.approx(0.0)),
    ]


def test_normalize_semantic_clamps_negative_to_zero() -> None:
    result = normalize_semantic([_semantic(distance=3.0)])

    assert result == [(("d", 0), pytest.approx(0.0))]


def test_normalize_semantic_empty_returns_empty_list() -> None:
    assert normalize_semantic([]) == []


def test_normalize_bm25_min_max_scales_into_unit_range() -> None:
    matches = [
        _bm25(score=1.0, doc_id="a", chunk_index=0),
        _bm25(score=3.0, doc_id="b", chunk_index=1),
        _bm25(score=5.0, doc_id="c", chunk_index=2),
    ]

    result = normalize_bm25(matches)

    assert result == [
        (("a", 0), pytest.approx(0.0)),
        (("b", 1), pytest.approx(0.5)),
        (("c", 2), pytest.approx(1.0)),
    ]


def test_normalize_bm25_single_match_returns_one() -> None:
    result = normalize_bm25([_bm25(score=42.5)])

    assert result == [(("d", 0), pytest.approx(1.0))]


def test_normalize_bm25_all_equal_scores_return_one() -> None:
    matches = [
        _bm25(score=2.5, doc_id="a", chunk_index=0),
        _bm25(score=2.5, doc_id="b", chunk_index=1),
        _bm25(score=2.5, doc_id="c", chunk_index=2),
    ]

    result = normalize_bm25(matches)

    assert result == [
        (("a", 0), pytest.approx(1.0)),
        (("b", 1), pytest.approx(1.0)),
        (("c", 2), pytest.approx(1.0)),
    ]


def test_normalize_bm25_empty_returns_empty_list() -> None:
    assert normalize_bm25([]) == []


def test_normalize_fuzzy_divides_by_hundred() -> None:
    matches = [
        _fuzzy(score=100.0, doc_id="a", chunk_index=0),
        _fuzzy(score=50.0, doc_id="b", chunk_index=1),
        _fuzzy(score=0.0, doc_id="c", chunk_index=2),
    ]

    result = normalize_fuzzy(matches)

    assert result == [
        (("a", 0), pytest.approx(1.0)),
        (("b", 1), pytest.approx(0.5)),
        (("c", 2), pytest.approx(0.0)),
    ]


def test_normalize_fuzzy_empty_returns_empty_list() -> None:
    assert normalize_fuzzy([]) == []
