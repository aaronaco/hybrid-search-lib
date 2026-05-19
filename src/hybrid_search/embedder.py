"""Local embedding model wrapper."""

from collections.abc import Iterable
from typing import Any, Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]: ...


def _sentence_transformer_class() -> type[Any]:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _to_float_vectors(vectors: Any) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [_to_float_list(vector) for vector in vectors]


class Embedder:
    """Wrap a local sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    @property
    def _loaded_model(self) -> Any:
        if self._model is None:
            self._model = _sentence_transformer_class()(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return _to_float_list(self._loaded_model.encode(text))

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return _to_float_vectors(self._loaded_model.encode(list(texts)))
