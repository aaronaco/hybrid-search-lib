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

Lifecycle behavior, ranking weights, persistence details, and runnable example
files are documented in later INIT-010 stories.
