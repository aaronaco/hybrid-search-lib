from pathlib import Path

from hybrid_search import HybridSearch


def test_add_feeds_internal_indices(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    
    search.add("doc-1", "title", "content")
    
    # Internal _bm25_index and _fuzzy_index are populated
    assert len(search._bm25_index._corpus_tokens) > 0
    assert len(search._fuzzy_index._indexed_texts) > 0
    
    # Verify vector index has the chunk
    collection = search._vector_index.collection
    assert collection.count() > 0


def test_delete_removes_from_internal_indices(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    
    search.add("doc-2", "title 2", "content 2")
    search.delete("doc-2")
    
    assert len(search._bm25_index._corpus_tokens) == 0
    assert len(search._fuzzy_index._indexed_texts) == 0
    
    collection = search._vector_index.collection
    assert collection.count() == 0


def test_empty_search_indices(tmp_path: Path) -> None:
    search = HybridSearch(storage_path=tmp_path)
    
    assert len(search._bm25_index._corpus_tokens) == 0
    assert len(search._fuzzy_index._indexed_texts) == 0
