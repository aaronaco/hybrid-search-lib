import pytest

from hybrid_search.bm25 import BM25Match
from hybrid_search.fuzzy import FuzzyMatch
from hybrid_search.index import SemanticMatch
from hybrid_search.ranker import rank
from hybrid_search.result import SearchResult


DEFAULT_WEIGHTS = {"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2}


def _semantic(
    distance: float,
    doc_id: str = "d",
    chunk_index: int = 0,
    title: str = "t",
    text: str = "x",
) -> SemanticMatch:
    return SemanticMatch(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title=title,
        text=text,
        distance=distance,
    )


def _bm25(
    score: float,
    doc_id: str = "d",
    chunk_index: int = 0,
    title: str = "t",
    text: str = "x",
) -> BM25Match:
    return BM25Match(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title=title,
        text=text,
        score=score,
    )


def _fuzzy(
    score: float,
    doc_id: str = "d",
    chunk_index: int = 0,
    title: str = "t",
    text: str = "x",
) -> FuzzyMatch:
    return FuzzyMatch(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title=title,
        text=text,
        score=score,
    )


def test_rank_returns_empty_when_all_layers_empty() -> None:
    assert rank([], [], [], DEFAULT_WEIGHTS, top_k=5) == []


def test_rank_deduplicates_multiple_chunks_per_doc_to_one_result() -> None:
    semantic = [
        _semantic(distance=0.0, doc_id="a", chunk_index=0, text="chunk-0"),
        _semantic(distance=0.0, doc_id="a", chunk_index=1, text="chunk-1"),
    ]

    result = rank(semantic, [], [], DEFAULT_WEIGHTS, top_k=5)

    assert len(result) == 1
    assert result[0].doc_id == "a"


def test_rank_winner_is_highest_final_score_chunk() -> None:
    semantic = [
        _semantic(distance=0.0, doc_id="a", chunk_index=0, text="winner-text"),
        _semantic(distance=0.0, doc_id="a", chunk_index=1, text="loser-text"),
    ]
    bm25 = [
        _bm25(score=5.0, doc_id="a", chunk_index=0, text="winner-text"),
        _bm25(score=1.0, doc_id="a", chunk_index=1, text="loser-text"),
    ]

    result = rank(semantic, bm25, [], DEFAULT_WEIGHTS, top_k=5)

    assert len(result) == 1
    assert result[0].doc_id == "a"
    assert result[0].matched_chunk == "winner-text"
    assert result[0].semantic_score == pytest.approx(1.0)
    assert result[0].bm25_score == pytest.approx(1.0)
    assert result[0].fuzzy_score == 0.0
    assert result[0].score == pytest.approx(0.4 * 1.0 + 0.4 * 1.0)


def test_rank_truncates_to_top_k() -> None:
    semantic = [
        _semantic(distance=0.0, doc_id="a"),
        _semantic(distance=0.2, doc_id="b"),
        _semantic(distance=0.4, doc_id="c"),
        _semantic(distance=0.6, doc_id="d"),
        _semantic(distance=0.8, doc_id="e"),
    ]

    result = rank(semantic, [], [], DEFAULT_WEIGHTS, top_k=3)

    assert len(result) == 3
    assert [r.doc_id for r in result] == ["a", "b", "c"]
    # Scores must be descending.
    assert result[0].score > result[1].score > result[2].score


def test_rank_returns_all_when_fewer_than_top_k() -> None:
    semantic = [
        _semantic(distance=0.4, doc_id="a"),
        _semantic(distance=0.0, doc_id="b"),
    ]

    result = rank(semantic, [], [], DEFAULT_WEIGHTS, top_k=10)

    assert len(result) == 2
    assert [r.doc_id for r in result] == ["b", "a"]
    assert result[0].score > result[1].score


def test_rank_excludes_zero_final_score_winners() -> None:
    # Only the semantic layer matched at distance=2.0 → normalized 0.0;
    # final_score = 0.4 * 0.0 = 0.0 → excluded.
    semantic = [_semantic(distance=2.0, doc_id="a")]

    result = rank(semantic, [], [], DEFAULT_WEIGHTS, top_k=5)

    assert result == []


def test_rank_stable_tie_break_preserves_first_seen() -> None:
    # Both docs hit only semantic at distance=0.0; final_score is bit-exact equal.
    # Insertion order in semantic list is doc_a first, so doc_a wins the tie.
    semantic = [
        _semantic(distance=0.0, doc_id="doc_a"),
        _semantic(distance=0.0, doc_id="doc_b"),
    ]

    result = rank(semantic, [], [], DEFAULT_WEIGHTS, top_k=5)

    assert [r.doc_id for r in result] == ["doc_a", "doc_b"]
    assert result[0].score == pytest.approx(result[1].score)


def test_rank_search_result_field_mapping() -> None:
    bm25 = [_bm25(score=1.0, doc_id="a", chunk_index=3, title="My Title", text="body")]

    result = rank([], bm25, [], DEFAULT_WEIGHTS, top_k=5)

    assert len(result) == 1
    assert isinstance(result[0], SearchResult)
    r = result[0]
    assert r.doc_id == "a"
    assert r.title == "My Title"
    assert r.matched_chunk == "body"
    assert r.semantic_score == 0.0
    assert r.bm25_score == pytest.approx(1.0)
    assert r.fuzzy_score == 0.0
    assert r.score == pytest.approx(0.4)


def test_rank_returns_top_k_zero_as_empty_list() -> None:
    # top_k == 0 is an internal-trust case; the public path rejects it.
    # rank() with top_k=0 must still produce [], not crash.
    semantic = [_semantic(distance=0.0, doc_id="a")]

    assert rank(semantic, [], [], DEFAULT_WEIGHTS, top_k=0) == []
