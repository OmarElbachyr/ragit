"""Tests for collection initialization and document metadata handling."""

import csv
import json
from pathlib import Path

import fitz
import pytest

from ragit.data import initialize_collection
from ragit.data.validation import CollectionValidationError


def create_pdf(path: Path, page_count: int) -> None:
    """Create a minimal PDF with the requested number of pages."""
    document = fitz.open()

    for _ in range(page_count):
        document.new_page()

    document.save(path)
    document.close()


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSON Lines records."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSON Lines records."""
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def write_qrels(
    path: Path,
    rows: list[tuple[int, int, int]],
) -> None:
    """Write canonical page-level qrels."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query_id", "page_id", "score"])
        writer.writerows(rows)


def test_initializes_collection_deterministically(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "b.pdf", 1)
    create_pdf(pdfs_path / "a.pdf", 2)

    collection = initialize_collection(pdfs_path)

    assert [document.doc_id for document in collection.documents] == [
        "a",
        "b",
    ]

    assert [
        (page.page_id, page.doc_id, page.page_number)
        for page in collection.pages
    ] == [
        (0, "a", 0),
        (1, "a", 1),
        (2, "b", 0),
    ]

    ragit_path = pdfs_path / ".ragit"

    assert (ragit_path / "corpus.jsonl").exists()
    assert (ragit_path / "documents_metadata.jsonl").exists()
    assert not (ragit_path / "queries.jsonl").exists()
    assert not (ragit_path / "qrels.tsv").exists()


def test_creates_default_document_metadata(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 3)

    initialize_collection(pdfs_path)

    metadata_path = (
        pdfs_path / ".ragit" / "documents_metadata.jsonl"
    )
    records = read_jsonl(metadata_path)

    assert records == [
        {
            "file_name": "report.pdf",
            "doc_id": "report",
            "local_path": str(pdfs_path / "report.pdf"),
            "page_number": 3,
        }
    ]


def test_reuses_existing_corpus_and_metadata(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 1)

    initialize_collection(pdfs_path)

    ragit_path = pdfs_path / ".ragit"
    corpus_path = ragit_path / "corpus.jsonl"
    metadata_path = ragit_path / "documents_metadata.jsonl"

    original_corpus = corpus_path.read_text(encoding="utf-8")

    metadata_record = read_jsonl(metadata_path)[0]
    metadata_record["doc_language"] = "english"
    metadata_record["doc_type"] = "annual_report"

    write_jsonl(metadata_path, [metadata_record])

    collection = initialize_collection(pdfs_path)

    assert corpus_path.read_text(encoding="utf-8") == original_corpus

    loaded_document = collection.documents[0]

    assert loaded_document.doc_id == "report"
    assert loaded_document.model_extra == {
        "doc_language": "english",
        "doc_type": "annual_report",
    }

def test_optional_metadata_extraction_is_not_implemented(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 1)

    with pytest.raises(
        NotImplementedError,
        match="Optional metadata extraction is not implemented yet",
    ):
        initialize_collection(
            pdfs_path=pdfs_path,
            extract_metadata=True,
        )

def test_imports_and_overwrites_queries_and_qrels(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 2)

    first_queries_path = tmp_path / "first_queries.jsonl"
    first_qrels_path = tmp_path / "first_qrels.tsv"

    write_jsonl(
        first_queries_path,
        [
            {
                "query_id": 0,
                "text": "First query",
                "language": "english",
            }
        ],
    )
    write_qrels(
        first_qrels_path,
        [(0, 0, 1)],
    )

    initialize_collection(
        pdfs_path=pdfs_path,
        queries_path=first_queries_path,
        qrels_path=first_qrels_path,
    )

    second_queries_path = tmp_path / "second_queries.jsonl"
    second_qrels_path = tmp_path / "second_qrels.tsv"

    write_jsonl(
        second_queries_path,
        [
            {
                "query_id": 0,
                "text": "Replacement query",
                "language": "english",
            },
            {
                "query_id": 1,
                "text": "Second query",
                "answer": "Optional answer",
            },
        ],
    )
    write_qrels(
        second_qrels_path,
        [
            (0, 1, 2),
            (1, 0, 1),
        ],
    )

    collection = initialize_collection(
        pdfs_path=pdfs_path,
        queries_path=second_queries_path,
        qrels_path=second_qrels_path,
    )

    assert [query.text for query in collection.queries] == [
        "Replacement query",
        "Second query",
    ]

    assert collection.queries[0].model_extra == {
        "language": "english"
    }
    assert collection.queries[1].model_extra == {
        "answer": "Optional answer"
    }

    assert [
        (qrel.query_id, qrel.page_id, qrel.score)
        for qrel in collection.qrels
    ] == [
        (0, 1, 2),
        (1, 0, 1),
    ]


def test_reuses_existing_queries_and_qrels(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 1)

    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    write_jsonl(
        queries_path,
        [{"query_id": 0, "text": "A query"}],
    )
    write_qrels(
        qrels_path,
        [(0, 0, 1)],
    )

    initialize_collection(
        pdfs_path=pdfs_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )

    collection = initialize_collection(pdfs_path)

    assert len(collection.queries) == 1
    assert collection.queries[0].text == "A query"
    assert len(collection.qrels) == 1


@pytest.mark.parametrize(
    ("qrel_row", "expected_message"),
    [
        ((1, 0, 1), "unknown query ID"),
        ((0, 99, 1), "unknown page ID"),
    ],
)
def test_rejects_invalid_qrel_reference(
    tmp_path: Path,
    qrel_row: tuple[int, int, int],
    expected_message: str,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 1)

    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    write_jsonl(
        queries_path,
        [{"query_id": 0, "text": "A query"}],
    )
    write_qrels(
        qrels_path,
        [qrel_row],
    )

    with pytest.raises(
        CollectionValidationError,
        match=expected_message,
    ):
        initialize_collection(
            pdfs_path=pdfs_path,
            queries_path=queries_path,
            qrels_path=qrels_path,
        )


def test_rejects_non_contiguous_query_ids(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 1)

    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    write_jsonl(
        queries_path,
        [
            {"query_id": 0, "text": "First"},
            {"query_id": 2, "text": "Third"},
        ],
    )
    write_qrels(
        qrels_path,
        [
            (0, 0, 1),
            (2, 0, 1),
        ],
    )

    with pytest.raises(
        CollectionValidationError,
        match="Query IDs must start at 0 and be contiguous",
    ):
        initialize_collection(
            pdfs_path=pdfs_path,
            queries_path=queries_path,
            qrels_path=qrels_path,
        )
def test_optional_metadata_extraction_is_not_implemented(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    create_pdf(pdfs_path / "report.pdf", 1)

    with pytest.raises(
        NotImplementedError,
        match="Optional metadata extraction is not implemented yet",
    ):
        initialize_collection(
            pdfs_path=pdfs_path,
            extract_metadata=True,
        )