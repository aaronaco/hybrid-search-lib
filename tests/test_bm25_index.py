import pytest

from hybrid_search import bm25 as bm25_module
from hybrid_search.bm25 import BM25Index, BM25Match
from hybrid_search.chunker import Chunk


class FakeBM25Okapi:
    """Test double that records its corpus and returns scripted scores."""

    instances: list["FakeBM25Okapi"] = []
    next_scores: list[float] | None = None

    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = [list(tokens) for tokens in corpus]
        self.get_scores_calls: list[list[str]] = []
        self.scripted_scores = FakeBM25Okapi.next_scores
        FakeBM25Okapi.next_scores = None
        FakeBM25Okapi.instances.append(self)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        self.get_scores_calls.append(list(query_tokens))
        if self.scripted_scores is not None:
            return list(self.scripted_scores)
        return [0.0] * len(self.corpus)


def use_fake_bm25_okapi(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeBM25Okapi.instances = []
    FakeBM25Okapi.next_scores = None
    monkeypatch.setattr(bm25_module, "_bm25_okapi_class", lambda: FakeBM25Okapi)


def test_construction_does_not_instantiate_okapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)

    BM25Index()

    assert FakeBM25Okapi.instances == []


def test_empty_search_returns_empty_list_without_initializing_okapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()

    matches = index.search("anything", top_k=5)

    assert matches == []
    assert FakeBM25Okapi.instances == []


def test_add_chunks_records_per_chunk_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    chunks = [
        Chunk(doc_id="doc-A", chunk_index=0, title="Hello", text="World, it's 2026!"),
        Chunk(doc_id="doc-B", chunk_index=2, title="Onboarding", text="docs/setup.md"),
    ]

    index.add_chunks(chunks)

    assert index._corpus_tokens == [
        ["hello", "world", "it", "s", "2026"],
        ["onboarding", "docs", "setup", "md"],
    ]
    assert index._chunk_keys == [("doc-A", 0), ("doc-B", 2)]
    assert index._metadata == {
        ("doc-A", 0): {"title": "Hello", "text": "World, it's 2026!"},
        ("doc-B", 2): {"title": "Onboarding", "text": "docs/setup.md"},
    }
    assert FakeBM25Okapi.instances == []


def test_add_chunks_empty_sequence_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()

    index.add_chunks([])

    assert index._corpus_tokens == []
    assert index._chunk_keys == []
    assert index._metadata == {}
    assert FakeBM25Okapi.instances == []


def _three_chunks() -> list[Chunk]:
    return [
        Chunk(doc_id="doc-A", chunk_index=0, title="Alpha", text="alpha body"),
        Chunk(doc_id="doc-B", chunk_index=0, title="Beta", text="beta body"),
        Chunk(doc_id="doc-C", chunk_index=0, title="Gamma", text="gamma body"),
    ]


def test_search_returns_bm25_matches_in_descending_score_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.1, 0.9, 0.5]

    matches = index.search("body", top_k=3)

    assert matches == [
        BM25Match(
            doc_id="doc-B",
            chunk_index=0,
            title="Beta",
            text="beta body",
            score=0.9,
        ),
        BM25Match(
            doc_id="doc-C",
            chunk_index=0,
            title="Gamma",
            text="gamma body",
            score=0.5,
        ),
        BM25Match(
            doc_id="doc-A",
            chunk_index=0,
            title="Alpha",
            text="alpha body",
            score=0.1,
        ),
    ]


def test_search_lazily_initializes_okapi_with_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.1, 0.2, 0.3]

    index.search("body", top_k=3)

    assert len(FakeBM25Okapi.instances) == 1
    assert FakeBM25Okapi.instances[0].corpus == [
        ["alpha", "alpha", "body"],
        ["beta", "beta", "body"],
        ["gamma", "gamma", "body"],
    ]
    assert FakeBM25Okapi.instances[0].get_scores_calls == [["body"]]


def test_search_truncates_to_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.1, 0.9, 0.5]

    matches = index.search("body", top_k=2)

    assert [m.doc_id for m in matches] == ["doc-B", "doc-C"]


def test_search_filters_out_zero_score_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.8, 0.0, 0.0]

    matches = index.search("body", top_k=5)

    assert len(matches) == 1
    assert matches[0].doc_id == "doc-A"
    assert matches[0].score == 0.8


def test_search_returns_empty_list_when_all_scores_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.0, 0.0, 0.0]

    matches = index.search("body", top_k=5)

    assert matches == []


def test_search_raises_when_top_k_is_not_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())

    with pytest.raises(ValueError):
        index.search("body", top_k=0)
    with pytest.raises(ValueError):
        index.search("body", top_k=-1)

    assert FakeBM25Okapi.instances == []


def test_search_raises_when_top_k_is_not_positive_on_empty_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()

    with pytest.raises(ValueError):
        index.search("body", top_k=0)
    with pytest.raises(ValueError):
        index.search("body", top_k=-1)

    assert FakeBM25Okapi.instances == []


def test_search_empty_query_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())

    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []
    assert FakeBM25Okapi.instances == []


def test_remove_document_drops_all_chunks_for_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(
        [
            Chunk(doc_id="doc-A", chunk_index=0, title="A0", text="a zero"),
            Chunk(doc_id="doc-A", chunk_index=1, title="A1", text="a one"),
            Chunk(doc_id="doc-B", chunk_index=0, title="B0", text="b zero"),
        ]
    )

    index.remove_document("doc-A")

    assert index._chunk_keys == [("doc-B", 0)]
    assert index._corpus_tokens == [["b0", "b", "zero"]]
    assert index._metadata == {("doc-B", 0): {"title": "B0", "text": "b zero"}}


def test_remove_document_invalidates_cached_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.1, 0.1, 0.1]
    index.search("body", top_k=3)
    assert len(FakeBM25Okapi.instances) == 1

    index.remove_document("doc-A")
    FakeBM25Okapi.next_scores = [0.1, 0.1]
    index.search("body", top_k=3)

    assert len(FakeBM25Okapi.instances) == 2


def test_remove_unknown_document_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.1, 0.1, 0.1]
    index.search("body", top_k=3)
    assert len(FakeBM25Okapi.instances) == 1
    snapshot_keys = list(index._chunk_keys)
    snapshot_tokens = [list(toks) for toks in index._corpus_tokens]

    index.remove_document("ghost")

    assert index._chunk_keys == snapshot_keys
    assert index._corpus_tokens == snapshot_tokens
    # Cache stayed warm; no second rebuild needed.
    FakeBM25Okapi.next_scores = [0.1, 0.1, 0.1]
    index.search("body", top_k=3)
    assert len(FakeBM25Okapi.instances) == 1


def test_add_chunks_invalidates_cached_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_bm25_okapi(monkeypatch)
    index = BM25Index()
    index.add_chunks(_three_chunks())
    FakeBM25Okapi.next_scores = [0.1, 0.1, 0.1]
    index.search("body", top_k=3)
    assert len(FakeBM25Okapi.instances) == 1

    index.add_chunks(
        [Chunk(doc_id="doc-D", chunk_index=0, title="Delta", text="delta body")]
    )
    FakeBM25Okapi.next_scores = [0.1, 0.1, 0.1, 0.1]
    index.search("body", top_k=3)

    assert len(FakeBM25Okapi.instances) == 2


def test_real_bm25_returns_matches_for_distinctive_term() -> None:
    index = BM25Index()
    index.add_chunks(
        [
            Chunk(doc_id="doc-1", chunk_index=0, title="T1", text="distinctive alpha"),
            Chunk(doc_id="doc-2", chunk_index=0, title="T2", text="distinctive beta"),
            Chunk(doc_id="doc-3", chunk_index=0, title="T3", text="distinctive gamma"),
        ]
    )

    matches = index.search("beta", top_k=3)

    assert len(matches) == 1
    assert matches[0].doc_id == "doc-2"


def test_real_bm25_remove_document_drops_matches() -> None:
    index = BM25Index()
    index.add_chunks(
        [
            Chunk(doc_id="doc-1", chunk_index=0, title="T1", text="common term here"),
            Chunk(doc_id="doc-2", chunk_index=0, title="T2", text="common term here"),
            Chunk(doc_id="doc-2", chunk_index=1, title="T3", text="another term"),
            Chunk(doc_id="doc-3", chunk_index=0, title="T4", text="filler"),
            Chunk(doc_id="doc-4", chunk_index=0, title="T5", text="filler"),
            Chunk(doc_id="doc-5", chunk_index=0, title="T6", text="filler"),
        ]
    )

    index.remove_document("doc-1")
    matches = index.search("common", top_k=10)

    assert len(matches) == 1
    assert matches[0].doc_id == "doc-2"
