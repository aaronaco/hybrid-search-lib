from collections.abc import Iterable
from pathlib import Path

from hybrid_search import HybridSearch
from hybrid_search import core as core_module
from hybrid_search.embedder import Embedder
from hybrid_search.index import SemanticMatch


class EmptyVectorIndex:
    def __init__(self, storage_path):
        self.storage_path = storage_path

    def list_chunks(self):
        return []


class TrackingEmbedder:
    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.embed_batch_calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [1.0, 0.0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        self.embed_batch_calls.append(text_list)
        return [[1.0, 0.0] for _ in text_list]


def test_constructor_accepts_custom_embedder_for_add(tmp_path: Path) -> None:
    embedder = TrackingEmbedder()
    search = HybridSearch(storage_path=tmp_path, embedder=embedder)

    search.add(
        doc_id="doc-1",
        title="Onboarding",
        content="alpha setup guide",
    )

    assert embedder.embed_batch_calls == [["Onboarding\nalpha setup guide"]]


def test_constructor_accepts_custom_embedder_for_query(tmp_path: Path) -> None:
    embedder = TrackingEmbedder()
    search = HybridSearch(storage_path=tmp_path, embedder=embedder)
    search._vector_index.query = lambda vector, top_k: [  # type: ignore[method-assign]
        SemanticMatch(
            doc_id="doc-1",
            chunk_index=0,
            title="Onboarding",
            text="alpha setup guide",
            distance=0.0,
        )
    ]

    results = search.query("alpha")

    assert embedder.embed_calls == ["alpha"]
    assert results[0].doc_id == "doc-1"


def test_default_constructor_still_uses_internal_embedder(monkeypatch) -> None:
    monkeypatch.setattr(core_module, "VectorIndex", EmptyVectorIndex)

    search = HybridSearch()

    assert isinstance(search._embedder, Embedder)
