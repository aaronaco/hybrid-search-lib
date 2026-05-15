import pytest

from hybrid_search.bm25 import BM25Match
from hybrid_search.fuzzy import FuzzyMatch
from hybrid_search.index import SemanticMatch
from hybrid_search.ranker import FusedChunk, fuse_candidates


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


def test_fuse_returns_empty_when_all_layers_empty() -> None:
    assert fuse_candidates([], [], [], DEFAULT_WEIGHTS) == []


def test_fuse_combines_three_overlapping_layers_with_default_weights() -> None:
    semantic = [_semantic(distance=0.5, doc_id="a", chunk_index=0)]
    bm25 = [_bm25(score=5.0, doc_id="a", chunk_index=0)]
    fuzzy = [_fuzzy(score=80.0, doc_id="a", chunk_index=0)]

    result = fuse_candidates(semantic, bm25, fuzzy, DEFAULT_WEIGHTS)

    assert len(result) == 1
    fused = result[0]
    assert fused.doc_id == "a"
    assert fused.chunk_index == 0
    assert fused.semantic_score == pytest.approx(0.75)
    assert fused.bm25_score == pytest.approx(1.0)
    assert fused.fuzzy_score == pytest.approx(0.8)
    assert fused.final_score == pytest.approx(0.4 * 0.75 + 0.4 * 1.0 + 0.2 * 0.8)


def test_fuse_zero_fills_missing_layer_components() -> None:
    semantic = [_semantic(distance=0.0, doc_id="a", chunk_index=0)]

    result = fuse_candidates(semantic, [], [], DEFAULT_WEIGHTS)

    assert len(result) == 1
    fused = result[0]
    assert fused.semantic_score == pytest.approx(1.0)
    assert fused.bm25_score == 0.0
    assert fused.fuzzy_score == 0.0
    assert fused.final_score == pytest.approx(0.4)


def test_fuse_respects_non_default_weights() -> None:
    semantic = [_semantic(distance=0.5, doc_id="a", chunk_index=0)]
    bm25 = [_bm25(score=5.0, doc_id="a", chunk_index=0)]
    fuzzy = [_fuzzy(score=80.0, doc_id="a", chunk_index=0)]
    weights = {"semantic": 0.7, "bm25": 0.2, "fuzzy": 0.1}

    result = fuse_candidates(semantic, bm25, fuzzy, weights)

    assert result[0].final_score == pytest.approx(0.7 * 0.75 + 0.2 * 1.0 + 0.1 * 0.8)


def test_fuse_union_emits_each_key_once() -> None:
    semantic = [_semantic(distance=0.0, doc_id="a", chunk_index=0)]
    bm25 = [
        _bm25(score=1.0, doc_id="a", chunk_index=0),
        _bm25(score=2.0, doc_id="b", chunk_index=1),
    ]
    fuzzy = [
        _fuzzy(score=100.0, doc_id="b", chunk_index=1),
        _fuzzy(score=50.0, doc_id="c", chunk_index=2),
    ]

    result = fuse_candidates(semantic, bm25, fuzzy, DEFAULT_WEIGHTS)

    assert len(result) == 3
    assert {(f.doc_id, f.chunk_index) for f in result} == {
        ("a", 0),
        ("b", 1),
        ("c", 2),
    }


def test_fuse_preserves_title_and_text_from_raw_matches() -> None:
    bm25 = [_bm25(score=1.0, doc_id="a", chunk_index=0, title="My Title", text="body here")]

    result = fuse_candidates([], bm25, [], DEFAULT_WEIGHTS)

    assert result[0].title == "My Title"
    assert result[0].text == "body here"


def test_fuse_emits_keys_in_deterministic_order() -> None:
    semantic = [
        _semantic(distance=0.0, doc_id="a", chunk_index=0),
        _semantic(distance=0.0, doc_id="b", chunk_index=0),
    ]
    bm25 = [
        _bm25(score=1.0, doc_id="c", chunk_index=0),
        _bm25(score=2.0, doc_id="a", chunk_index=0),
    ]
    fuzzy = [
        _fuzzy(score=100.0, doc_id="d", chunk_index=0),
        _fuzzy(score=50.0, doc_id="b", chunk_index=0),
    ]

    result = fuse_candidates(semantic, bm25, fuzzy, DEFAULT_WEIGHTS)

    assert [(f.doc_id, f.chunk_index) for f in result] == [
        ("a", 0),
        ("b", 0),
        ("c", 0),
        ("d", 0),
    ]


def test_fuse_returns_fused_chunk_instances() -> None:
    bm25 = [_bm25(score=1.0)]

    result = fuse_candidates([], bm25, [], DEFAULT_WEIGHTS)

    assert isinstance(result[0], FusedChunk)
