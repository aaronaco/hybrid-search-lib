"""Public search result value type."""

from dataclasses import dataclass


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
