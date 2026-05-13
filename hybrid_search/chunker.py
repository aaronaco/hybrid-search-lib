"""Document chunk representation and text assembly."""

from dataclasses import dataclass

_SHORT_DOCUMENT_THRESHOLD_WORDS = 200


@dataclass
class Chunk:
    """Internal chunk value object carrying document metadata."""

    doc_id: str
    chunk_index: int
    title: str
    text: str


def assemble_searchable_text(title: str, content: str) -> str:
    parts = [part for part in (title, content) if part]
    return "\n".join(parts)


def chunk_document(
    doc_id: str,
    title: str,
    content: str,
    chunk_size: int,
    chunk_overlap: float,
) -> list[Chunk]:
    text = assemble_searchable_text(title, content)
    words = text.split()
    if len(words) <= _SHORT_DOCUMENT_THRESHOLD_WORDS:
        return [Chunk(doc_id=doc_id, chunk_index=0, title=title, text=text)]

    overlap_words = max(1, int(chunk_size * chunk_overlap))
    stride = max(1, chunk_size - overlap_words)
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=idx,
                title=title,
                text=" ".join(words[start:end]),
            )
        )
        idx += 1
        if end == len(words):
            break
        start += stride
    return chunks
