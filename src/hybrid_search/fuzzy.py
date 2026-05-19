"""Internal Fuzzy retrieval wrapper."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hybrid_search.chunker import Chunk


def _rapidfuzz_process() -> Any:
    from rapidfuzz import process

    return process


def _rapidfuzz_fuzz() -> Any:
    from rapidfuzz import fuzz

    return fuzz


@dataclass
class FuzzyMatch:
    """Internal Fuzzy result carrying chunk identity and raw score."""

    doc_id: str
    chunk_index: int
    title: str
    text: str
    score: float


class FuzzyIndex:
    """Per-chunk fuzzy lookup using rapidfuzz."""

    def __init__(self) -> None:
        self._indexed_texts: list[str] = []
        self._chunk_keys: list[tuple[str, int]] = []
        self._metadata: dict[tuple[str, int], dict[str, str]] = {}

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            indexed_text = f"{chunk.title}\n{chunk.text}"
            key = (chunk.doc_id, chunk.chunk_index)
            self._indexed_texts.append(indexed_text)
            self._chunk_keys.append(key)
            self._metadata[key] = {"title": chunk.title, "text": chunk.text}

    def remove_document(self, doc_id: str) -> None:
        kept_texts: list[str] = []
        kept_keys: list[tuple[str, int]] = []
        removed = False
        for text, key in zip(self._indexed_texts, self._chunk_keys):
            if key[0] == doc_id:
                self._metadata.pop(key, None)
                removed = True
                continue
            kept_texts.append(text)
            kept_keys.append(key)
        if removed:
            self._indexed_texts = kept_texts
            self._chunk_keys = kept_keys

    def search(self, query: str, top_k: int) -> list[FuzzyMatch]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive: got {top_k}")
        if not self._indexed_texts:
            return []
        if not query.strip():
            return []

        results = _rapidfuzz_process().extract(
            query,
            self._indexed_texts,
            scorer=_rapidfuzz_fuzz().partial_ratio,
            limit=top_k,
        )

        matches = []
        for _, score, idx in results:
            key = self._chunk_keys[idx]
            matches.append(
                FuzzyMatch(
                    doc_id=key[0],
                    chunk_index=key[1],
                    title=self._metadata[key]["title"],
                    text=self._metadata[key]["text"],
                    score=float(score),
                )
            )
        return matches
