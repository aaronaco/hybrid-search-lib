from hybrid_search import HybridSearch


def test_new_instance_has_empty_document_registry() -> None:
    search = HybridSearch()

    assert search._document_ids == set()


def test_register_document_id_makes_it_discoverable() -> None:
    search = HybridSearch()

    search._register_document_id("doc-1")

    assert search._has_document_id("doc-1") is True


def test_unregister_document_id_removes_it() -> None:
    search = HybridSearch()
    search._register_document_id("doc-1")

    search._unregister_document_id("doc-1")

    assert search._has_document_id("doc-1") is False
