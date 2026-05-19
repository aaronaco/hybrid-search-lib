from collections.abc import Iterable
from pathlib import Path

from hybrid_search import HybridSearch
from hybrid_search import core as core_module
from hybrid_search import embedder as embedder_module
from hybrid_search.embedder import DEFAULT_EMBEDDING_MODEL, Embedder
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


class RecordingEmbedder:
    model_names: list[str] = []

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model_names.append(model_name)

    def embed(self, text: str) -> list[float]:
        return [1.0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def use_recording_embedder(monkeypatch) -> None:
    RecordingEmbedder.model_names = []
    monkeypatch.setattr(core_module, "VectorIndex", EmptyVectorIndex)
    monkeypatch.setattr(core_module, "Embedder", RecordingEmbedder)


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


def test_constructor_passes_embedding_model_to_default_embedder(monkeypatch) -> None:
    use_recording_embedder(monkeypatch)

    search = HybridSearch(embedding_model="some-model")

    assert isinstance(search._embedder, RecordingEmbedder)
    assert search._embedder.model_name == "some-model"
    assert RecordingEmbedder.model_names == ["some-model"]


def test_constructor_uses_default_embedding_model_when_unspecified(monkeypatch) -> None:
    use_recording_embedder(monkeypatch)

    search = HybridSearch()

    assert isinstance(search._embedder, RecordingEmbedder)
    assert search._embedder.model_name == DEFAULT_EMBEDDING_MODEL
    assert RecordingEmbedder.model_names == [DEFAULT_EMBEDDING_MODEL]


def test_constructor_passes_local_embedding_model_path_unchanged(
    monkeypatch,
    tmp_path: Path,
) -> None:
    use_recording_embedder(monkeypatch)
    model_path = str(tmp_path / "local-model")

    search = HybridSearch(embedding_model=model_path)

    assert isinstance(search._embedder, RecordingEmbedder)
    assert search._embedder.model_name == model_path
    assert RecordingEmbedder.model_names == [model_path]


def test_embedding_model_constructor_does_not_load_sentence_transformer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(core_module, "VectorIndex", EmptyVectorIndex)

    def fail_if_loaded():
        raise AssertionError("sentence-transformers should not load during construction")

    monkeypatch.setattr(embedder_module, "_sentence_transformer_class", fail_if_loaded)

    search = HybridSearch(storage_path=tmp_path, embedding_model="some-model")

    assert isinstance(search._embedder, Embedder)
    assert search._embedder.model_name == "some-model"
