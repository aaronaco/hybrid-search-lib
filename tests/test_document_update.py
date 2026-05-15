from pathlib import Path

from hybrid_search import HybridSearch


def test_update_replaces_previous_indexed_state(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search.add("d", "old t", "old c")

    search.update("d", "new t", "new c")

    assert search._documents["d"] == {"title": "new t", "content": "new c"}


def test_update_keeps_doc_id_registered(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search.add("d", "old t", "old c")

    search.update("d", "new t", "new c")

    assert search._has_document_id("d") is True


def test_update_accepts_empty_title_with_non_empty_content(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search.add("d", "t", "c")

    search.update("d", "", "still has content")

    assert search._documents["d"]["title"] == ""
    assert search._documents["d"]["content"] == "still has content"
