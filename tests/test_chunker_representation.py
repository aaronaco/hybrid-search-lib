from hybrid_search.chunker import Chunk, assemble_searchable_text


def test_chunk_carries_document_metadata_fields() -> None:
    chunk = Chunk(doc_id="d", chunk_index=2, title="t", text="x")

    assert chunk.doc_id == "d"
    assert chunk.chunk_index == 2
    assert chunk.title == "t"
    assert chunk.text == "x"


def test_assemble_searchable_text_includes_title_and_content() -> None:
    result = assemble_searchable_text("a title", "some content")

    assert "a title" in result
    assert "some content" in result


def test_assemble_searchable_text_preserves_content_when_title_missing() -> None:
    assert assemble_searchable_text("", "just content") == "just content"


def test_assemble_searchable_text_preserves_title_when_content_missing() -> None:
    assert assemble_searchable_text("just title", "") == "just title"


def test_assemble_searchable_text_returns_empty_string_when_both_missing() -> None:
    assert assemble_searchable_text("", "") == ""
