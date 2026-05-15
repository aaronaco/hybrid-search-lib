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


def test_delete_unknown_doc_id_does_not_mutate_internal_indexes() -> None:
    search = HybridSearch()
    search.add("d", "t", "c")

    bm25_keys_snapshot = list(search._bm25_index._chunk_keys)
    bm25_tokens_snapshot = [list(toks) for toks in search._bm25_index._corpus_tokens]
    fuzzy_keys_snapshot = list(search._fuzzy_index._chunk_keys)
    fuzzy_texts_snapshot = list(search._fuzzy_index._indexed_texts)

    with pytest.raises(KeyError):
        search.delete("ghost")

    assert search._bm25_index._chunk_keys == bm25_keys_snapshot
    assert search._bm25_index._corpus_tokens == bm25_tokens_snapshot
    assert search._fuzzy_index._chunk_keys == fuzzy_keys_snapshot
    assert search._fuzzy_index._indexed_texts == fuzzy_texts_snapshot
