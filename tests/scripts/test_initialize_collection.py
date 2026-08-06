"""Manually test initialize_collection on real benchmark data."""

from pathlib import Path

from ragit.data import initialize_collection


DATA_PATH = Path("data/vidore_v3_finance_en")
PDFS_PATH = DATA_PATH / "pdfs"

CORPUS_PATH = DATA_PATH / "corpus.jsonl"
DOCUMENTS_METADATA_PATH = DATA_PATH / "documents_metadata.jsonl"
QUERIES_PATH = DATA_PATH / "queries.jsonl"
QRELS_PATH = DATA_PATH / "qrels.tsv"


# Change this value to run one scenario at a time.
SCENARIO = 1


def print_collection(collection) -> None:
    print("\nCollection initialized successfully")
    print(f"Ragit directory: {collection.ragit_path}")
    print(f"Documents:       {len(collection.documents)}")
    print(f"Pages:           {len(collection.pages)}")
    print(f"Queries:         {len(collection.queries)}")
    print(f"Qrels:           {len(collection.qrels)}")

    print("\nCanonical files:")
    for path in sorted(collection.ragit_path.iterdir()):
        if path.is_file():
            print(f"- {path.name}")

    print("\nFirst document:")
    print(collection.documents[0].model_dump())

    print("\nFirst page:")
    print(collection.pages[0].model_dump())

    if collection.queries:
        print("\nFirst query:")
        print(collection.queries[0].model_dump())

    if collection.qrels:
        print("\nFirst qrel:")
        print(collection.qrels[0].model_dump())


if SCENARIO == 1:
    # Existing benchmark:
    # import corpus, metadata, queries, and qrels while preserving their IDs.
    collection = initialize_collection(
        pdfs_path=PDFS_PATH,
        corpus_path=CORPUS_PATH,
        documents_metadata_path=DOCUMENTS_METADATA_PATH,
        queries_path=QUERIES_PATH,
        qrels_path=QRELS_PATH,
    )
    print_collection(collection)


elif SCENARIO == 2:
    # Reuse all files already stored under .ragit/.
    # Run Scenario 1 first.
    collection = initialize_collection(
        pdfs_path=PDFS_PATH,
    )
    print_collection(collection)


elif SCENARIO == 3:
    # Raw PDF collection:
    # generate corpus and minimum metadata automatically.
    #
    # Only use this after deleting .ragit/, because otherwise existing
    # benchmark files will be reused.
    collection = initialize_collection(
        pdfs_path=PDFS_PATH,
    )
    print_collection(collection)


elif SCENARIO == 4:
    # Optional metadata extraction is intentionally unavailable for now.
    # Expected result: NotImplementedError.
    collection = initialize_collection(
        pdfs_path=PDFS_PATH,
        extract_metadata=True,
    )
    print_collection(collection)


else:
    raise ValueError(f"Unknown scenario: {SCENARIO}")