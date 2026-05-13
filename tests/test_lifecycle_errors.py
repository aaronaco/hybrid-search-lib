import pytest

from hybrid_search import HybridSearch


def test_add_duplicate_doc_id_raises_value_error() -> None:
    search = HybridSearch()
    search.add("d", "t", "c")

    with pytest.raises(ValueError):
        search.add("d", "t", "c")


def test_add_duplicate_does_not_corrupt_prior_state() -> None:
    search = HybridSearch()
    search.add("d", "t", "c")

    with pytest.raises(ValueError):
        search.add("d", "different t", "different c")

    assert search._documents["d"] == {"title": "t", "content": "c"}


def test_update_unknown_doc_id_raises_key_error() -> None:
    search = HybridSearch()

    with pytest.raises(KeyError):
        search.update("nope", "t", "c")


def test_delete_unknown_doc_id_raises_key_error() -> None:
    search = HybridSearch()

    with pytest.raises(KeyError):
        search.delete("nope")
