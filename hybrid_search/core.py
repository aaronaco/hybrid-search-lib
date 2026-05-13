"""Public facade for local hybrid search."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

_DEFAULT_STORAGE_PATH = Path("~/.hybrid_search")
_DEFAULT_WEIGHTS = MappingProxyType({"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2})


@dataclass
class SearchResult:
    """Search result with final and component scores."""

    doc_id: str
    title: str
    score: float
    matched_chunk: str
    semantic_score: float
    bm25_score: float
    fuzzy_score: float


class HybridSearch:
    """Configure a local hybrid search index."""

    def __init__(
        self,
        storage_path: str | Path | None = None,
        chunk_size: int = 256,
        chunk_overlap: float = 0.15,
        weights: Mapping[str, float] | None = None,
        top_k: int = 5,
    ) -> None:
        configured_path = _DEFAULT_STORAGE_PATH if storage_path is None else Path(storage_path)

        self.storage_path = configured_path.expanduser().resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.weights = dict(_DEFAULT_WEIGHTS if weights is None else weights)
        self.top_k = top_k
