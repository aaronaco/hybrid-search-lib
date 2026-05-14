"""Integration tests against real ChromaDB persistence."""

from pathlib import Path

from hybrid_search.chunker import Chunk
from hybrid_search.index import SemanticMatch, VectorIndex


def test_vector_index_creates_storage_directory_on_first_use(tmp_path: Path) -> None:
    storage_path = tmp_path / "chroma-store"
    assert not storage_path.exists()

    index = VectorIndex(storage_path)
    assert not storage_path.exists()

    chunk = Chunk(doc_id="d", chunk_index=0, title="t", text="x")
    index.add_chunks([chunk], [[0.1, 0.2]])

    assert storage_path.exists()
    assert storage_path.is_dir()
    assert any(storage_path.iterdir())


def test_indexed_chunks_are_retrievable_after_reinstantiation(tmp_path: Path) -> None:
    storage_path = tmp_path / "chroma-store"
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    vec_c = [0.7071, 0.7071]
    chunks = [
        Chunk(doc_id="doc-A", chunk_index=0, title="A", text="alpha chunk"),
        Chunk(doc_id="doc-B", chunk_index=0, title="B", text="beta chunk"),
        Chunk(doc_id="doc-C", chunk_index=0, title="C", text="gamma chunk"),
    ]

    writer = VectorIndex(storage_path)
    writer.add_chunks(chunks, [vec_a, vec_b, vec_c])
    del writer

    reader = VectorIndex(storage_path)
    matches = reader.query(vec_a, top_k=3)

    assert len(matches) == 3
    assert {(m.doc_id, m.chunk_index) for m in matches} == {
        (c.doc_id, c.chunk_index) for c in chunks
    }
    assert matches[0].doc_id == "doc-A"
    assert matches[0].chunk_index == 0


def test_chunk_metadata_fields_are_preserved_after_reinstantiation(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    chunk = Chunk(
        doc_id="docs/onboarding.md",
        chunk_index=7,
        title="Onboarding Guide",
        text="Welcome to the project.",
    )
    embedding = [0.5, 0.5]

    writer = VectorIndex(storage_path)
    writer.add_chunks([chunk], [embedding])
    del writer

    reader = VectorIndex(storage_path)
    matches = reader.query(embedding, top_k=1)

    assert len(matches) == 1
    match = matches[0]
    assert match.doc_id == "docs/onboarding.md"
    assert match.chunk_index == 7
    assert match.title == "Onboarding Guide"
    assert match.text == "Welcome to the project."
    assert isinstance(match.distance, float)
    assert match.distance >= 0.0
    assert isinstance(match, SemanticMatch)


def test_deleted_document_chunks_stay_removed_after_reinstantiation(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chroma-store"
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    vec_c = [0.7071, 0.7071]
    chunks = [
        Chunk(doc_id="doc-A", chunk_index=0, title="A", text="a0"),
        Chunk(doc_id="doc-A", chunk_index=1, title="A", text="a1"),
        Chunk(doc_id="doc-B", chunk_index=0, title="B", text="b0"),
    ]

    writer = VectorIndex(storage_path)
    writer.add_chunks(chunks, [vec_a, vec_b, vec_c])
    writer.delete_document("doc-A")
    del writer

    reader = VectorIndex(storage_path)
    matches = reader.query(vec_a, top_k=10)

    assert len(matches) == 1
    assert matches[0].doc_id == "doc-B"
    assert matches[0].chunk_index == 0
