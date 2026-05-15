from pathlib import Path
from tempfile import TemporaryDirectory

from hybrid_search import HybridSearch


def main() -> None:
    with TemporaryDirectory(prefix="hybrid-search-example-") as directory:
        storage_path = Path(directory) / "index"
        search = HybridSearch(storage_path=storage_path)

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
            print(f"doc_id: {result.doc_id}")
            print(f"title: {result.title}")
            print(f"score: {result.score:.3f}")
            print(f"matched_chunk: {result.matched_chunk}")
            print(
                "components: "
                f"semantic={result.semantic_score:.3f}, "
                f"bm25={result.bm25_score:.3f}, "
                f"fuzzy={result.fuzzy_score:.3f}"
            )
            print()


if __name__ == "__main__":
    main()
