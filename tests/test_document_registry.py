from pathlib import Path

from hybrid_search import HybridSearch


def test_new_instance_has_empty_document_registry(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)

    assert search._document_ids == set()


def test_register_document_id_makes_it_discoverable(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)

    search._register_document_id("doc-1")

    assert search._has_document_id("doc-1") is True


def test_unregister_document_id_removes_it(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    search._register_document_id("doc-1")

    search._unregister_document_id("doc-1")

    assert search._has_document_id("doc-1") is False
