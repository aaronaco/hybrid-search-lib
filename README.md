# hybrid-search-lib

`hybrid-search-lib` is a local Python library for hybrid search over text
documents. It combines semantic retrieval, BM25 keyword matching, and fuzzy
matching behind one importable `HybridSearch` facade.

The library is designed for local-first applications. Normal indexing and
querying use local storage and local embedding models rather than a managed
search service.

## Requirements

- Python `>=3.11,<3.14`
- A project environment managed by your preferred Python tool
- Enough local disk space for the ChromaDB index and the default
  sentence-transformers model cache

## Installation

Install the current v1 package directly from GitHub:

```powershell
uv add git+https://github.com/<owner>/hybrid-search-lib.git
```

For editable local development from a clone:

```powershell
uv sync
```

## Quickstart

```python
from hybrid_search import HybridSearch


search = HybridSearch(storage_path="./my_index")

search.add(
    doc_id="doc-1",
    title="Onboarding Guide",
    content="Welcome to the alpha project. This document explains setup.",
)
search.add(
    doc_id="doc-2",
    title="Architecture Notes",
    content="The library stores local vectors and combines multiple signals.",
)

results = search.query("alpha setup")

for result in results:
    print(result.doc_id)
    print(result.title)
    print(result.score)
    print(result.matched_chunk)
    print(result.semantic_score, result.bm25_score, result.fuzzy_score)
```

`query()` returns a list of `SearchResult` objects. Each result includes:

- `doc_id`
- `title`
- `score`
- `matched_chunk`
- `semantic_score`
- `bm25_score`
- `fuzzy_score`

`score` is the final ranking score used to order returned documents.
`semantic_score`, `bm25_score`, and `fuzzy_score` expose the public component
scores that contributed to that final score. `matched_chunk` contains the text
chunk that produced the best document-level result.

## Lifecycle

Use caller-managed document IDs for lifecycle operations. `add()` indexes a new
document, `update()` replaces an existing document's title and content, and
`delete()` removes an existing document.

```python
search.add(
    doc_id="doc-3",
    title="Release Checklist",
    content="Review tests, update documentation, and publish the package.",
)

search.update(
    doc_id="doc-3",
    title="Release Checklist",
    content="Review tests, update documentation, tag the release, and publish.",
)

search.delete("doc-3")
```

## Error Behavior

The public API uses explicit exceptions for lifecycle mistakes:

- `add()` with a duplicate `doc_id` raises `ValueError`.
- `update()` with an unknown `doc_id` raises `KeyError`.
- `delete()` with an unknown `doc_id` raises `KeyError`.
- `query("")` and whitespace-only queries return `[]`.
- `query(..., top_k=0)` and any other non-positive `top_k` raise
  `ValueError`.
- `query(..., weights=...)` raises `ValueError` when weights are missing a
  required key, include an extra key, include a negative value, or sum to zero.

## First-Run Notes

The default embedder uses the local
`sentence-transformers/all-MiniLM-L6-v2` model through sentence-transformers.
The first indexing or query workflow that needs embeddings may initialize the
model and populate the local model cache.

Indexes are stored under `storage_path`. Reusing the same path lets a new
`HybridSearch` instance recover persisted search metadata across restarts.

## Development Checks

Use the uv-managed project environment for validation:

```powershell
uv run pytest
uv run ruff check
uv run mypy hybrid_search tests
```

Ranking weight tuning, persistence details, and runnable example files are
documented in later INIT-010 stories.
