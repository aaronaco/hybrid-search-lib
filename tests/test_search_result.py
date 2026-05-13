from hybrid_search import SearchResult


EXPECTED_SEARCH_RESULT_FIELDS = [
    "doc_id",
    "title",
    "score",
    "matched_chunk",
    "semantic_score",
    "bm25_score",
    "fuzzy_score",
]


def test_search_result_fields_exist() -> None:
    result = SearchResult(
        doc_id="doc-1",
        title="Example",
        score=0.9,
        matched_chunk="matched text",
        semantic_score=0.8,
        bm25_score=0.7,
        fuzzy_score=0.6,
    )

    for field_name in EXPECTED_SEARCH_RESULT_FIELDS:
        assert hasattr(result, field_name)


def test_search_result_preserves_supplied_values() -> None:
    result = SearchResult(
        doc_id="doc-1",
        title="Example",
        score=0.9,
        matched_chunk="matched text",
        semantic_score=0.8,
        bm25_score=0.7,
        fuzzy_score=0.6,
    )

    assert result.doc_id == "doc-1"
    assert result.title == "Example"
    assert result.score == 0.9
    assert result.matched_chunk == "matched text"
    assert result.semantic_score == 0.8
    assert result.bm25_score == 0.7
    assert result.fuzzy_score == 0.6


def test_search_result_annotations_match_public_contract() -> None:
    assert list(SearchResult.__annotations__) == EXPECTED_SEARCH_RESULT_FIELDS
