"""Internal ChromaDB vector index wrapper."""

from pathlib import Path
from collections.abc import Callable
from typing import Any

DEFAULT_COLLECTION_NAME = "hybrid_search_chunks"


def _persistent_client_class() -> Callable[..., Any]:
    from chromadb import PersistentClient

    return PersistentClient


class VectorIndex:
    """Lazily create a local persistent ChromaDB collection."""

    def __init__(
        self,
        storage_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _persistent_client_class()(path=str(self.storage_path))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        return self._collection
