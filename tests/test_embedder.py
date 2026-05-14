from hybrid_search import embedder as embedder_module
from hybrid_search.embedder import DEFAULT_EMBEDDING_MODEL, Embedder


class ArrayLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSentenceTransformer:
    instances: list[str] = []

    def __init__(self, model_name: str) -> None:
        self.instances.append(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            return ArrayLike([1, 2.5, 3])
        return ArrayLike([[float(index), float(index + 1)] for index, _ in enumerate(texts)])


def use_fake_sentence_transformer(monkeypatch):
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(
        embedder_module,
        "_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )


def test_embed_returns_list_of_floats(monkeypatch) -> None:
    use_fake_sentence_transformer(monkeypatch)

    result = Embedder().embed("hello")

    assert result == [1.0, 2.5, 3.0]
    assert all(isinstance(value, float) for value in result)


def test_embed_batch_returns_one_vector_per_input(monkeypatch) -> None:
    use_fake_sentence_transformer(monkeypatch)

    result = Embedder().embed_batch(["a", "b"])

    assert result == [[0.0, 1.0], [1.0, 2.0]]
    assert len(result) == 2
    assert all(isinstance(vector, list) for vector in result)


def test_repeated_calls_reuse_loaded_model(monkeypatch) -> None:
    use_fake_sentence_transformer(monkeypatch)
    embedder = Embedder()

    embedder.embed("one")
    embedder.embed("two")
    embedder.embed_batch(["three", "four"])

    assert FakeSentenceTransformer.instances == [DEFAULT_EMBEDDING_MODEL]


def test_default_model_name_is_all_minilm_l6_v2() -> None:
    assert Embedder().model_name == "sentence-transformers/all-MiniLM-L6-v2"
