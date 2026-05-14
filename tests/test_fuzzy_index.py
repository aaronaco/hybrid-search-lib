import typing

import pytest

from hybrid_search import fuzzy as fuzzy_module
from hybrid_search.fuzzy import FuzzyIndex, FuzzyMatch
from hybrid_search.chunker import Chunk


class FakeRapidFuzzFuzz:
    @staticmethod
    def partial_ratio(*args, **kwargs):
        return "fake_partial_ratio_scorer"


class FakeRapidFuzzProcess:
    instances: list["FakeRapidFuzzProcess"] = []
    next_extract_results: list[tuple[str, float, int]] | None = None

    def __init__(self) -> None:
        self.extract_calls: list[dict] = []
        FakeRapidFuzzProcess.instances.append(self)

    def extract(
        self, query: str, choices: list[str], scorer: typing.Any = None, limit: int = 5
    ) -> list[tuple[str, float, int]]:
        self.extract_calls.append(
            {
                "query": query,
                "choices": list(choices),
                "scorer": scorer,
                "limit": limit,
            }
        )
        if FakeRapidFuzzProcess.next_extract_results is not None:
            return FakeRapidFuzzProcess.next_extract_results
        return []


def use_fake_rapidfuzz(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRapidFuzzProcess.instances = []
    FakeRapidFuzzProcess.next_extract_results = None
    monkeypatch.setattr(
        fuzzy_module, "_rapidfuzz_process", lambda: FakeRapidFuzzProcess()
    )
    monkeypatch.setattr(fuzzy_module, "_rapidfuzz_fuzz", lambda: FakeRapidFuzzFuzz)


def test_construction_does_not_instantiate_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)

    FuzzyIndex()

    assert FakeRapidFuzzProcess.instances == []


def test_empty_search_returns_empty_list_without_calling_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()

    matches = index.search("anything", top_k=5)

    assert matches == []
    assert FakeRapidFuzzProcess.instances == []


def test_add_chunks_records_per_chunk_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()
    chunks = [
        Chunk(doc_id="doc-A", chunk_index=0, title="Hello", text="World"),
        Chunk(doc_id="doc-B", chunk_index=2, title="Test", text="Text"),
    ]

    index.add_chunks(chunks)

    assert index._indexed_texts == ["Hello\nWorld", "Test\nText"]
    assert index._chunk_keys == [("doc-A", 0), ("doc-B", 2)]
    assert index._metadata == {
        ("doc-A", 0): {"title": "Hello", "text": "World"},
        ("doc-B", 2): {"title": "Test", "text": "Text"},
    }
    assert FakeRapidFuzzProcess.instances == []


def test_add_chunks_empty_sequence_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()

    index.add_chunks([])

    assert index._indexed_texts == []
    assert index._chunk_keys == []
    assert index._metadata == {}
    assert FakeRapidFuzzProcess.instances == []


def _three_chunks() -> list[Chunk]:
    return [
        Chunk(doc_id="doc-A", chunk_index=0, title="Alpha", text="alpha body"),
        Chunk(doc_id="doc-B", chunk_index=0, title="Beta", text="beta body"),
        Chunk(doc_id="doc-C", chunk_index=0, title="Gamma", text="gamma body"),
    ]


def test_remove_document_drops_all_chunks_for_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()
    index.add_chunks(
        [
            Chunk(doc_id="doc-A", chunk_index=0, title="A0", text="a zero"),
            Chunk(doc_id="doc-A", chunk_index=1, title="A1", text="a one"),
            Chunk(doc_id="doc-B", chunk_index=0, title="B0", text="b zero"),
        ]
    )

    index.remove_document("doc-A")

    assert index._chunk_keys == [("doc-B", 0)]
    assert index._indexed_texts == ["B0\nb zero"]
    assert index._metadata == {("doc-B", 0): {"title": "B0", "text": "b zero"}}


def test_search_calls_extract_and_maps_results_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()
    index.add_chunks(_three_chunks())
    FakeRapidFuzzProcess.next_extract_results = [
        ("Beta\nbeta body", 90.0, 1),
        ("Alpha\nalpha body", 45.5, 0),
    ]

    matches = index.search("beta", top_k=2)

    assert len(matches) == 2
    assert matches[0] == FuzzyMatch("doc-B", 0, "Beta", "beta body", 90.0)
    assert matches[1] == FuzzyMatch("doc-A", 0, "Alpha", "alpha body", 45.5)

    process_instance = FakeRapidFuzzProcess.instances[0]
    assert len(process_instance.extract_calls) == 1
    call = process_instance.extract_calls[0]
    assert call["query"] == "beta"
    assert call["choices"] == [
        "Alpha\nalpha body",
        "Beta\nbeta body",
        "Gamma\ngamma body",
    ]
    assert call["scorer"] == FakeRapidFuzzFuzz.partial_ratio
    assert call["limit"] == 2


def test_search_raises_when_top_k_is_not_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()
    index.add_chunks(_three_chunks())

    with pytest.raises(ValueError):
        index.search("query", top_k=0)
    with pytest.raises(ValueError):
        index.search("query", top_k=-1)

    assert FakeRapidFuzzProcess.instances == []


def test_search_empty_query_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_rapidfuzz(monkeypatch)
    index = FuzzyIndex()
    index.add_chunks(_three_chunks())

    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []
    assert FakeRapidFuzzProcess.instances == []


def test_real_fuzzy_returns_matches_for_typo() -> None:
    index = FuzzyIndex()
    index.add_chunks(
        [
            Chunk(doc_id="doc-1", chunk_index=0, title="T1", text="distinctive alpha"),
            Chunk(doc_id="doc-2", chunk_index=0, title="T2", text="distinctive beta"),
            Chunk(doc_id="doc-3", chunk_index=0, title="T3", text="distinctive gamma"),
        ]
    )

    matches = index.search("bta", top_k=3)

    assert len(matches) > 0
    assert matches[0].doc_id == "doc-2"
    assert matches[0].score > 50.0


def test_real_fuzzy_returns_matches_for_substring() -> None:
    index = FuzzyIndex()
    index.add_chunks(
        [
            Chunk(
                doc_id="doc-1",
                chunk_index=0,
                title="T1",
                text="some long document text here",
            ),
            Chunk(
                doc_id="doc-2",
                chunk_index=0,
                title="T2",
                text="another different piece of content",
            ),
        ]
    )

    matches = index.search("document text", top_k=3)

    assert len(matches) > 0
    assert matches[0].doc_id == "doc-1"
    assert matches[0].score == 100.00
