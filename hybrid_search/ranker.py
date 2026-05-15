"""Internal score normalization and ranking helpers."""

from collections.abc import Sequence

from hybrid_search.bm25 import BM25Match
from hybrid_search.fuzzy import FuzzyMatch
from hybrid_search.index import SemanticMatch

ChunkKey = tuple[str, int]
NormalizedCandidate = tuple[ChunkKey, float]


def normalize_semantic(
    matches: Sequence[SemanticMatch],
) -> list[NormalizedCandidate]:
    if not matches:
        return []
    return [
        ((m.doc_id, m.chunk_index), max(0.0, 1.0 - (m.distance / 2.0)))
        for m in matches
    ]


def normalize_bm25(
    matches: Sequence[BM25Match],
) -> list[NormalizedCandidate]:
    if not matches:
        return []
    scores = [m.score for m in matches]
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [((m.doc_id, m.chunk_index), 1.0) for m in matches]
    span = hi - lo
    return [
        ((m.doc_id, m.chunk_index), (m.score - lo) / span)
        for m in matches
    ]


def normalize_fuzzy(
    matches: Sequence[FuzzyMatch],
) -> list[NormalizedCandidate]:
    if not matches:
        return []
    return [
        ((m.doc_id, m.chunk_index), m.score / 100.0)
        for m in matches
    ]
