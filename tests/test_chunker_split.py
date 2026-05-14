from hybrid_search.chunker import chunk_document


def test_short_document_returns_single_chunk() -> None:
    out = chunk_document("d", "t", " ".join(["w"] * 50), 256, 0.15)

    assert len(out) == 1
    assert out[0].chunk_index == 0
    assert out[0].doc_id == "d"
    assert out[0].title == "t"


def test_title_only_document_returns_title_chunk() -> None:
    out = chunk_document("d", "only title", "", 256, 0.15)

    assert len(out) == 1
    assert out[0].text == "only title"
    assert out[0].title == "only title"
    assert out[0].doc_id == "d"
    assert out[0].chunk_index == 0


def test_content_only_document_returns_content_chunk() -> None:
    out = chunk_document("d", "", "only content", 256, 0.15)

    assert len(out) == 1
    assert out[0].text == "only content"
    assert out[0].title == ""
    assert out[0].doc_id == "d"
    assert out[0].chunk_index == 0


def test_both_empty_document_returns_deterministic_empty_chunk() -> None:
    out = chunk_document("d", "", "", 256, 0.15)

    assert len(out) == 1
    assert out[0].text == ""
    assert out[0].title == ""
    assert out[0].doc_id == "d"
    assert out[0].chunk_index == 0


def test_document_at_threshold_returns_single_chunk() -> None:
    # 200 words with empty title -> exactly 200 words total, still one chunk
    out = chunk_document("d", "", " ".join(["w"] * 200), 256, 0.15)

    assert len(out) == 1


def test_long_document_returns_multiple_ordered_chunks() -> None:
    out = chunk_document(
        "d", "", " ".join([f"w{i}" for i in range(600)]), 256, 0.15
    )

    assert len(out) >= 2
    assert [c.chunk_index for c in out] == list(range(len(out)))


def test_adjacent_chunks_overlap_with_trailing_context() -> None:
    chunk_size = 256
    chunk_overlap = 0.15
    overlap_words = int(chunk_size * chunk_overlap)
    out = chunk_document(
        "d", "", " ".join([f"w{i}" for i in range(600)]), chunk_size, chunk_overlap
    )

    # The trailing `overlap_words` of chunk N must equal the leading
    # `overlap_words` of chunk N+1 — this is the contract overlap encodes.
    assert (
        out[0].text.split()[-overlap_words:]
        == out[1].text.split()[:overlap_words]
    )


def test_chunk_metadata_is_preserved_on_long_split() -> None:
    out = chunk_document(
        "doc-x", "the title", " ".join([f"w{i}" for i in range(600)]), 256, 0.15
    )

    assert all(c.doc_id == "doc-x" for c in out)
    assert all(c.title == "the title" for c in out)
