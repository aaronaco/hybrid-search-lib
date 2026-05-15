from pathlib import Path

from hybrid_search import HybridSearch


def test_add_registers_unique_doc_id(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)

    search.add("doc-1", "a title", "some content")

    assert search._has_document_id("doc-1") is True


def test_add_accepts_title_and_content(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)

    search.add("doc-1", "a title", "some content")

    assert search._documents["doc-1"] == {"title": "a title", "content": "some content"}


def test_add_accepts_empty_content_with_non_empty_title(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)

    search.add("doc-2", "only title", "")

    assert search._has_document_id("doc-2") is True
    assert search._documents["doc-2"]["content"] == ""
