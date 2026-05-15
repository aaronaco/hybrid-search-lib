from pathlib import Path

from hybrid_search import HybridSearch


def test_delete_unregisters_doc_id(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search.add("d", "t", "c")

    search.delete("d")

    assert search._has_document_id("d") is False


def test_delete_removes_stored_document(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search.add("d", "t", "c")

    search.delete("d")

    assert "d" not in search._documents


def test_delete_allows_subsequent_re_add_without_duplicate_state(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search.add("d", "t1", "c1")
    search.delete("d")

    search.add("d", "t2", "c2")

    assert search._has_document_id("d") is True
    assert search._documents["d"] == {"title": "t2", "content": "c2"}
