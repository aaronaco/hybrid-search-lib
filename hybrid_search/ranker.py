"""Internal score normalization and ranking helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hybrid_search.bm25 import BM25Match
from hybrid_search.fuzzy import FuzzyMatch
from hybrid_search.index import SemanticMatch

ChunkKey = tuple[str, int]
NormalizedCandidate = tuple[ChunkKey, float]


@dataclass
class FusedChunk:
    """Internal per-chunk fused score record."""

    doc_id: str
    chunk_index: int
    title: str
    text: str
    semantic_score: float
    bm25_score: float
    fuzzy_score: float
    final_score: float


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


def fuse_candidates(
    semantic: Sequence[SemanticMatch],
    bm25: Sequence[BM25Match],
    fuzzy: Sequence[FuzzyMatch],
    weights: Mapping[str, float],
) -> list[FusedChunk]:
    if not semantic and not bm25 and not fuzzy:
        return []

    sem_dict = dict(normalize_semantic(semantic))
    bm25_dict = dict(normalize_bm25(bm25))
    fuzz_dict = dict(normalize_fuzzy(fuzzy))

    metadata: dict[ChunkKey, tuple[str, str]] = {}
    ordered_keys: dict[ChunkKey, None] = {}
    for match_list in (semantic, bm25, fuzzy):
        for m in match_list:
            key = (m.doc_id, m.chunk_index)
            if key not in metadata:
                metadata[key] = (m.title, m.text)
            ordered_keys.setdefault(key)

    w_s = weights["semantic"]
    w_b = weights["bm25"]
    w_f = weights["fuzzy"]

    fused: list[FusedChunk] = []
    for key in ordered_keys:
        s = sem_dict.get(key, 0.0)
        b = bm25_dict.get(key, 0.0)
        f = fuzz_dict.get(key, 0.0)
        title, text = metadata[key]
        fused.append(
            FusedChunk(
                doc_id=key[0],
                chunk_index=key[1],
                title=title,
                text=text,
                semantic_score=s,
                bm25_score=b,
                fuzzy_score=f,
                final_score=w_s * s + w_b * b + w_f * f,
            )
        )
    return fused
