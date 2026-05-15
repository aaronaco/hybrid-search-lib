import ast
from pathlib import Path


EXAMPLE_PATH = Path("examples/basic_usage.py")


def _example_tree() -> ast.Module:
    return ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_basic_usage_example_compiles() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    compile(source, str(EXAMPLE_PATH), "exec")


def test_basic_usage_example_uses_only_public_hybrid_search_imports() -> None:
    tree = _example_tree()
    imported_names: list[str] = []
    internal_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "hybrid_search":
                imported_names.extend(alias.name for alias in node.names)
            elif node.module and node.module.startswith("hybrid_search."):
                internal_imports.append(node.module)
        elif isinstance(node, ast.Import):
            internal_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("hybrid_search.")
            )

    assert imported_names == ["HybridSearch"]
    assert internal_imports == []


def test_basic_usage_example_uses_temporary_storage() -> None:
    tree = _example_tree()

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TemporaryDirectory"
    ]

    assert len(calls) == 1
    assert any(
        keyword.arg == "prefix"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "hybrid-search-example-"
        for keyword in calls[0].keywords
    )
