# Contributing

Thanks for considering a contribution to `hybrid-search-lib`. This guide covers the validation gate, the public behavior contract, and the examples policy.

## Validation

Run the full validation triad before submitting changes:

```powershell
uv run pytest
uv run ruff check
uv run mypy src/hybrid_search tests
```

All three must pass.

## Public Behavior Contract

`tests/test_behavior_contract.py` is the canonical contract surface for the public `HybridSearch` and `SearchResult` API. Any change to documented public behavior, or any newly documented public behavior, must come with a corresponding test in that file. Per-feature unit tests continue to cover internal state at finer granularity, but the behavior contract suite is what readers and reviewers treat as the public-API source of truth.

## Examples

Example files live under `examples/`. Each example must remain runnable via:

```powershell
uv run examples/<file>.py
```

Examples must use only the public API, must not require cloud services or non-local infrastructure, and should keep storage isolated (for example, via `tempfile.TemporaryDirectory`) so they do not leave artifacts in the project root.
