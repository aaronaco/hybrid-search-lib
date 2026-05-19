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
- `query()` raises `ValueError` when the instance is configured with a
  non-positive `top_k`.
- `query()` raises `ValueError` when the instance is configured with malformed
  `weights`: missing required keys, extra keys, negative values, or a
  non-positive sum.

## Runnable Example

Run the basic example from a local clone with uv:

```powershell
uv run examples/basic_usage.py
```

The example uses a temporary storage directory, so it does not leave index
files in the project root. The first run may initialize the local embedding
model and populate the sentence-transformers cache.

## Configuration

`HybridSearch` exposes the main tuning controls through its constructor:

```python
search = HybridSearch(
    storage_path="./my_index",
    chunk_size=256,
    chunk_overlap=0.15,
    weights={"semantic": 0.4, "bm25": 0.4, "fuzzy": 0.2},
    top_k=5,
)
```

- `storage_path` controls where the local index is stored. The default is
  `~/.hybrid_search`.
- `chunk_size` is an approximate word count per chunk. The default is `256`.
- `chunk_overlap` controls overlap between adjacent chunks. The default is
  `0.15`.
- `weights` must contain exactly `semantic`, `bm25`, and `fuzzy` keys with
  non-negative values that sum to a positive number.
- `top_k` controls the maximum number of document results returned by
  `query()`. The default is `5`.

## Ranking and Tuning

`score` is the weighted sum of the public component scores:

```text
score = semantic_weight * semantic_score
      + bm25_weight * bm25_score
      + fuzzy_weight * fuzzy_score
```

Increase `weights["semantic"]` to favor semantic similarity, increase
`weights["bm25"]` to favor exact keyword overlap, and increase
`weights["fuzzy"]` to favor typo-tolerant or partial matches. Component scores
are useful for inspection and tuning, but exact relevance depends on the corpus
and query.

`top_k` sets the maximum number of document results returned. During query
processing, the library gathers a larger candidate pool with
`max(top_k * 4, 20)` before returning the best `top_k` document results.

## Persistence and Restart Behavior

Indexes are stored under `storage_path`. Reusing the same path lets a new
`HybridSearch` instance recover persisted search metadata across restarts.
Startup recovery rebuilds the derived keyword and fuzzy search state from the
persisted metadata, so exact, semantic, and typo-tolerant signals can contribute
after a restart without caller-managed reload code.

Use an explicit `storage_path` for application data you want to keep. Temporary
paths are useful for examples and tests, but their indexes disappear when the
temporary directory is removed.

## Local Model Behavior

The default embedder uses the local
`sentence-transformers/all-MiniLM-L6-v2` model through sentence-transformers.
The first indexing or query workflow that needs embeddings may initialize the
model and populate the local model cache.

Normal indexing and querying use local storage and local model execution. The
library does not require a managed cloud search service for these workflows.

## Development Checks

Use the uv-managed project environment for validation:

```powershell
uv run pytest
uv run ruff check
uv run mypy src/hybrid_search tests
```

## Contributing

See `CONTRIBUTING.md` for the validation gate, the public behavior contract
location, and the examples policy.
