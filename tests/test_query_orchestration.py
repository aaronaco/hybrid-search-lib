from pathlib import Path

import pytest

from hybrid_search import HybridSearch, SearchResult
from hybrid_search.bm25 import BM25Match
from hybrid_search.fuzzy import FuzzyMatch
from hybrid_search.index import SemanticMatch


class FakeEmbedder:
    def __init__(self) -> None:
        self.embed_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.0]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []
        self.next_results: list[SemanticMatch] = []

    def query(self, vector, top_k: int) -> list[SemanticMatch]:
        self.query_calls.append({"vector": list(vector), "top_k": top_k})
        return list(self.next_results)


class FakeBM25Index:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []
        self.next_results: list[BM25Match] = []

    def search(self, query: str, top_k: int) -> list[BM25Match]:
        self.search_calls.append({"query": query, "top_k": top_k})
        return list(self.next_results)


class FakeFuzzyIndex:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []
        self.next_results: list[FuzzyMatch] = []

    def search(self, query: str, top_k: int) -> list[FuzzyMatch]:
        self.search_calls.append({"query": query, "top_k": top_k})
        return list(self.next_results)


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


def _make_search(
    tmp_path: Path,
    **kwargs,
) -> tuple[HybridSearch, FakeEmbedder, FakeVectorIndex, FakeBM25Index, FakeFuzzyIndex]:
    search = HybridSearch(storage_path=tmp_path, **kwargs)
    emb = FakeEmbedder()
    vec = FakeVectorIndex()
    bm = FakeBM25Index()
    fz = FakeFuzzyIndex()
    search._embedder = emb  # type: ignore[assignment]
    search._vector_index = vec  # type: ignore[assignment]
    search._bm25_index = bm  # type: ignore[assignment]
    search._fuzzy_index = fz  # type: ignore[assignment]
    return search, emb, vec, bm, fz


def test_query_returns_search_result_list(tmp_path: Path) -> None:
    search, _emb, vec, bm, fz = _make_search(tmp_path)
    vec.next_results = [_semantic(distance=0.5, doc_id="a", text="vector text")]
    bm.next_results = [_bm25(score=1.0, doc_id="a", text="vector text")]
    fz.next_results = [_fuzzy(score=80.0, doc_id="a", text="vector text")]

    result = search.query("my text")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], SearchResult)
    assert result[0].doc_id == "a"


def test_query_calls_each_layer_exactly_once(tmp_path: Path) -> None:
    search, _emb, vec, bm, fz = _make_search(tmp_path)
    vec.next_results = [_semantic(distance=0.0, doc_id="a")]
    bm.next_results = [_bm25(score=1.0, doc_id="b")]
    fz.next_results = [_fuzzy(score=50.0, doc_id="c")]

    search.query("anything")

    assert len(vec.query_calls) == 1
    assert len(bm.search_calls) == 1
    assert len(fz.search_calls) == 1


def test_query_calls_embedder_once_with_query_text(tmp_path: Path) -> None:
    search, emb, _vec, _bm, _fz = _make_search(tmp_path)

    search.query("my search text")

    assert emb.embed_calls == ["my search text"]


def test_query_passes_candidate_size_to_each_layer(tmp_path: Path) -> None:
    search, _emb, vec, bm, fz = _make_search(tmp_path, top_k=10)

    search.query("anything")

    assert vec.query_calls[0]["top_k"] == 40
    assert bm.search_calls[0]["top_k"] == 40
    assert fz.search_calls[0]["top_k"] == 40


def test_query_passes_candidate_size_floor_when_top_k_is_small(tmp_path: Path) -> None:
    search, _emb, vec, bm, fz = _make_search(tmp_path, top_k=1)

    search.query("anything")

    assert vec.query_calls[0]["top_k"] == 20
    assert bm.search_calls[0]["top_k"] == 20
    assert fz.search_calls[0]["top_k"] == 20


def test_query_score_is_weighted_sum_of_components_default_weights(
    tmp_path: Path,
) -> None:
    search, _emb, vec, bm, fz = _make_search(tmp_path)
    vec.next_results = [_semantic(distance=0.5, doc_id="a", text="t")]
    bm.next_results = [_bm25(score=5.0, doc_id="a", text="t")]
    fz.next_results = [_fuzzy(score=80.0, doc_id="a", text="t")]

    result = search.query("anything")

    expected = 0.4 * 0.75 + 0.4 * 1.0 + 0.2 * 0.8
    assert result[0].semantic_score == pytest.approx(0.75)
    assert result[0].bm25_score == pytest.approx(1.0)
    assert result[0].fuzzy_score == pytest.approx(0.8)
    assert result[0].score == pytest.approx(expected)


def test_query_respects_non_default_weights(tmp_path: Path) -> None:
    search, _emb, vec, bm, fz = _make_search(
        tmp_path,
        weights={"semantic": 0.7, "bm25": 0.2, "fuzzy": 0.1},
    )
    vec.next_results = [_semantic(distance=0.5, doc_id="a", text="t")]
    bm.next_results = [_bm25(score=5.0, doc_id="a", text="t")]
    fz.next_results = [_fuzzy(score=80.0, doc_id="a", text="t")]

    result = search.query("anything")

    expected = 0.7 * 0.75 + 0.2 * 1.0 + 0.1 * 0.8
    assert result[0].score == pytest.approx(expected)


def test_query_truncates_to_instance_top_k(tmp_path: Path) -> None:
    search, _emb, vec, _bm, _fz = _make_search(tmp_path, top_k=3)
    vec.next_results = [
        _semantic(distance=0.0, doc_id="a"),
        _semantic(distance=0.2, doc_id="b"),
        _semantic(distance=0.4, doc_id="c"),
        _semantic(distance=0.6, doc_id="d"),
        _semantic(distance=0.8, doc_id="e"),
    ]

    result = search.query("anything")

    assert len(result) == 3
    assert [r.doc_id for r in result] == ["a", "b", "c"]
    assert result[0].score > result[1].score > result[2].score


def test_query_uses_text_as_query_string_for_bm25_and_fuzzy(tmp_path: Path) -> None:
    search, _emb, _vec, bm, fz = _make_search(tmp_path)

    search.query("my text")

    assert bm.search_calls[0]["query"] == "my text"
    assert fz.search_calls[0]["query"] == "my text"


def test_query_uses_embedder_vector_for_semantic_layer(tmp_path: Path) -> None:
    search, _emb, vec, _bm, _fz = _make_search(tmp_path)

    search.query("my text")

    assert vec.query_calls[0]["vector"] == [0.0]
