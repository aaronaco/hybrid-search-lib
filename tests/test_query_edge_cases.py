from pathlib import Path

import pytest

from hybrid_search import HybridSearch
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


def _assert_no_layer_called(
    emb: FakeEmbedder,
    vec: FakeVectorIndex,
    bm: FakeBM25Index,
    fz: FakeFuzzyIndex,
) -> None:
    assert emb.embed_calls == []
    assert vec.query_calls == []
    assert bm.search_calls == []
    assert fz.search_calls == []


def test_query_on_empty_index_returns_empty_list(tmp_path: Path) -> None:
    search, _emb, _vec, _bm, _fz = _make_search(tmp_path)

    # All four fakes default to empty next_results — simulates empty corpus.
    assert search.query("anything") == []


def test_query_empty_text_returns_empty_list_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search, emb, vec, bm, fz = _make_search(tmp_path)

    assert search.query("") == []
    _assert_no_layer_called(emb, vec, bm, fz)


def test_query_whitespace_text_returns_empty_list_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search, emb, vec, bm, fz = _make_search(tmp_path)

    assert search.query("   ") == []
    assert search.query("\t\n ") == []
    _assert_no_layer_called(emb, vec, bm, fz)


def test_query_raises_when_top_k_is_not_positive_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search_zero, emb_z, vec_z, bm_z, fz_z = _make_search(tmp_path, top_k=0)
    with pytest.raises(ValueError):
        search_zero.query("anything")
    _assert_no_layer_called(emb_z, vec_z, bm_z, fz_z)

    search_neg, emb_n, vec_n, bm_n, fz_n = _make_search(tmp_path, top_k=-1)
    with pytest.raises(ValueError):
        search_neg.query("anything")
    _assert_no_layer_called(emb_n, vec_n, bm_n, fz_n)


def test_query_raises_when_weights_missing_key_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search, emb, vec, bm, fz = _make_search(
        tmp_path,
        weights={"semantic": 0.5, "bm25": 0.5},
    )

    with pytest.raises(ValueError):
        search.query("anything")
    _assert_no_layer_called(emb, vec, bm, fz)


def test_query_raises_when_weights_has_extra_key_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search, emb, vec, bm, fz = _make_search(
        tmp_path,
        weights={"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2, "extra": 0.1},
    )

    with pytest.raises(ValueError):
        search.query("anything")
    _assert_no_layer_called(emb, vec, bm, fz)


def test_query_raises_when_weights_has_negative_value_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search, emb, vec, bm, fz = _make_search(
        tmp_path,
        weights={"semantic": -0.1, "bm25": 0.5, "fuzzy": 0.6},
    )

    with pytest.raises(ValueError):
        search.query("anything")
    _assert_no_layer_called(emb, vec, bm, fz)


def test_query_raises_when_weights_sum_to_zero_and_no_layer_called(
    tmp_path: Path,
) -> None:
    search, emb, vec, bm, fz = _make_search(
        tmp_path,
        weights={"semantic": 0.0, "bm25": 0.0, "fuzzy": 0.0},
    )

    with pytest.raises(ValueError):
        search.query("anything")
    _assert_no_layer_called(emb, vec, bm, fz)


def test_query_returns_empty_list_when_all_fused_scores_are_zero(
    tmp_path: Path,
) -> None:
    search, _emb, vec, _bm, _fz = _make_search(tmp_path)
    # distance=2.0 normalizes to max(0, 1 - 2.0/2.0) = 0.0 → final_score 0.0
    # → excluded by rank()'s zero-score filter.
    vec.next_results = [
        SemanticMatch(doc_id="a", chunk_index=0, title="t", text="x", distance=2.0)
    ]

    assert search.query("anything") == []
