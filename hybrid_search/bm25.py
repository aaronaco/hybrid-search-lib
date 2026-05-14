"""Internal BM25 keyword retrieval wrapper."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from hybrid_search.chunker import Chunk


def _bm25_okapi_class() -> Callable[..., Any]:
    from rank_bm25 import BM25Okapi

    return BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Match:
    """Internal BM25 result carrying chunk identity and raw score."""

    doc_id: str
    chunk_index: int
    title: str
    text: str
    score: float


class BM25Index:
    """Per-chunk BM25 corpus with lazy `BM25Okapi` (re)build on search."""

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._chunk_keys: list[tuple[str, int]] = []
        self._metadata: dict[tuple[str, int], dict[str, str]] = {}
        self._bm25: Any | None = None

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            indexed_text = f"{chunk.title}\n{chunk.text}"
            tokens = _tokenize(indexed_text)
            key = (chunk.doc_id, chunk.chunk_index)
            self._corpus_tokens.append(tokens)
            self._chunk_keys.append(key)
            self._metadata[key] = {"title": chunk.title, "text": chunk.text}
        self._bm25 = None

    def remove_document(self, doc_id: str) -> None:
        kept_tokens: list[list[str]] = []
        kept_keys: list[tuple[str, int]] = []
        removed = False
        for tokens, key in zip(self._corpus_tokens, self._chunk_keys):
            if key[0] == doc_id:
                self._metadata.pop(key, None)
                removed = True
                continue
            kept_tokens.append(tokens)
            kept_keys.append(key)
        if removed:
            self._corpus_tokens = kept_tokens
            self._chunk_keys = kept_keys
            self._bm25 = None

    def search(self, query: str, top_k: int) -> list[BM25Match]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive: got {top_k}")
        if not self._corpus_tokens:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        if self._bm25 is None:
            self._bm25 = _bm25_okapi_class()(self._corpus_tokens)
        scores = self._bm25.get_scores(query_tokens)
        scored = [
            (float(score), key)
            for score, key in zip(scores, self._chunk_keys)
            if score > 0
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            BM25Match(
                doc_id=key[0],
                chunk_index=key[1],
                title=self._metadata[key]["title"],
                text=self._metadata[key]["text"],
                score=score,
            )
            for score, key in scored[:top_k]
        ]
