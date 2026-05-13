import hybrid_search
from hybrid_search import HybridSearch


def test_package_imports() -> None:
    assert hybrid_search.__name__ == "hybrid_search"
    assert HybridSearch.__name__ == "HybridSearch"
