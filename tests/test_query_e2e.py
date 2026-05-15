"""End-to-end smoke test for HybridSearch.query.

Uses the real BM25Index and FuzzyIndex against a tiny in-memory corpus.
The semantic layer and embedder are faked to avoid loading ChromaDB and
sentence-transformers — those are exercised by their own dedicated test
modules. The point of this test is to verify the orchestration pipeline
end-to-end through real keyword/fuzzy retrieval and the ranker.
"""

from pathlib import Path

from hybrid_search import HybridSearch, SearchResult
from hybrid_search.index import SemanticMatch


class FakeEmbedder:
    """Fake implementing both Embedder methods used by HybridSearch.add and query."""

    def embed(self, text: str) -> list[float]:
        return [0.0]

    def embed_batch(self, texts) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeVectorIndex:
    """Fake whose add_chunks/delete_document are no-ops and query returns empty."""

    def add_chunks(self, chunks, embeddings) -> None:
        pass

    def query(self, vector, top_k: int) -> list[SemanticMatch]:
        return []

    def delete_document(self, doc_id: str) -> None:
        pass


def test_query_e2e_returns_ranked_dedup_results_with_real_bm25_and_fuzzy(
    tmp_path: Path,
) -> None:
    search = HybridSearch(storage_path=tmp_path, top_k=3)
    # Replace embedder + semantic layer with fakes BEFORE calling add(),
    # so add() does not try to load sentence-transformers or ChromaDB.
    search._embedder = FakeEmbedder()  # type: ignore[assignment]
    search._vector_index = FakeVectorIndex()  # type: ignore[assignment]

    search.add(
        doc_id="doc-1",
        title="Python guide",
        content="A tutorial on Python lists and dicts",
    )
    search.add(
        doc_id="doc-2",
        title="Rust intro",
        content="An introduction to Rust ownership",
    )
    search.add(
        doc_id="doc-3",
        title="JS notes",
        content="JavaScript notes on closures",
    )

    results = search.query("python lists")

    assert len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert {r.doc_id for r in results} <= {"doc-1", "doc-2", "doc-3"}
    # doc-1 has both "python" and "lists" as exact BM25 hits; expect rank 1.
    assert results[0].doc_id == "doc-1"
    # Document-level dedup invariant: no duplicate doc_ids in result list.
    assert len({r.doc_id for r in results}) == len(results)
    # Component scores are reconstructible (semantic is 0 because layer faked empty).
    assert results[0].semantic_score == 0.0
    assert results[0].bm25_score > 0.0
    assert results[0].score > 0.0
