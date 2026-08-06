"""Initialization and loading of canonical RAGit collections."""

import csv
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragit.data.metadata import (
    create_default_metadata,
    load_metadata,
    write_metadata,
)
from ragit.data.models import Collection, Page, Qrel, Query
from ragit.data.validation import (
    CollectionInitializationError,
    CollectionValidationError,
    read_pdf_page_count,
    validate_collection,
    validate_pdf_directory,
)


CORPUS_FILE = "corpus.jsonl"
DOCUMENTS_FILE = "documents_metadata.jsonl"
QUERIES_FILE = "queries.jsonl"
QRELS_FILE = "qrels.tsv"


def initialize_collection(
    pdfs_path: str | Path,
    corpus_path: str | Path | None = None,
    documents_metadata_path: str | Path | None = None,
    queries_path: str | Path | None = None,
    qrels_path: str | Path | None = None,
    extract_metadata: bool = False,
) -> Collection:
    """Initialize, validate, and load a canonical RAGit collection.

    Explicitly supplied files replace their canonical copies under
    ``<pdfs_path>/.ragit/``.

    When corpus or document metadata files are not supplied and do not
    already exist, they are generated from the PDF directory.
    """
    pdfs_path = Path(pdfs_path)
    pdf_files = validate_pdf_directory(pdfs_path)

    if (queries_path is None) != (qrels_path is None):
        raise CollectionInitializationError(
            "queries_path and qrels_path must be supplied together."
        )

    if extract_metadata:
        raise NotImplementedError(
            "Optional metadata extraction is not implemented yet."
        )

    ragit_path = pdfs_path / ".ragit"
    ragit_path.mkdir(parents=True, exist_ok=True)

    canonical_corpus_path = ragit_path / CORPUS_FILE
    canonical_metadata_path = ragit_path / DOCUMENTS_FILE
    canonical_queries_path = ragit_path / QUERIES_FILE
    canonical_qrels_path = ragit_path / QRELS_FILE

    # Explicitly supplied files always replace canonical copies.
    if corpus_path is not None:
        copy_file(
            source=Path(corpus_path),
            destination=canonical_corpus_path,
        )

    if documents_metadata_path is not None:
        copy_file(
            source=Path(documents_metadata_path),
            destination=canonical_metadata_path,
        )

    if queries_path is not None and qrels_path is not None:
        copy_file(
            source=Path(queries_path),
            destination=canonical_queries_path,
        )
        copy_file(
            source=Path(qrels_path),
            destination=canonical_qrels_path,
        )

    # Generate only missing collection files.
    corpus_missing = not canonical_corpus_path.exists()
    metadata_missing = not canonical_metadata_path.exists()

    if corpus_missing or metadata_missing:
        page_counts = {
            pdf_path.stem: read_pdf_page_count(pdf_path)
            for pdf_path in pdf_files
        }

        if corpus_missing:
            pages = create_corpus(
                pdf_files=pdf_files,
                page_counts=page_counts,
            )
            write_jsonl(
                canonical_corpus_path,
                (page.model_dump() for page in pages),
            )

        if metadata_missing:
            documents = create_default_metadata(
                pdf_files=pdf_files,
                page_counts=page_counts,
            )
            write_metadata(
                canonical_metadata_path,
                documents,
            )

    if canonical_queries_path.exists() != canonical_qrels_path.exists():
        raise CollectionValidationError(
            "Canonical queries.jsonl and qrels.tsv must both exist "
            "or both be absent."
        )

    return load_collection(pdfs_path)


def load_collection(
    pdfs_path: str | Path,
) -> Collection:
    """Load and validate an initialized RAGit collection."""
    pdfs_path = Path(pdfs_path)
    pdf_files = validate_pdf_directory(pdfs_path)

    ragit_path = pdfs_path / ".ragit"

    corpus_path = ragit_path / CORPUS_FILE
    metadata_path = ragit_path / DOCUMENTS_FILE
    queries_path = ragit_path / QUERIES_FILE
    qrels_path = ragit_path / QRELS_FILE

    if not corpus_path.exists():
        raise CollectionInitializationError(
            f"Missing canonical corpus file: {corpus_path}"
        )

    if not metadata_path.exists():
        raise CollectionInitializationError(
            f"Missing canonical metadata file: {metadata_path}"
        )

    try:
        documents = load_metadata(metadata_path)

        pages = [
            Page.model_validate(record)
            for record in read_jsonl(corpus_path)
        ]

        if queries_path.exists() and qrels_path.exists():
            queries = [
                Query.model_validate(record)
                for record in read_jsonl(queries_path)
            ]
            qrels = read_qrels(qrels_path)

        elif not queries_path.exists() and not qrels_path.exists():
            queries = []
            qrels = []

        else:
            raise CollectionValidationError(
                "Canonical queries.jsonl and qrels.tsv must both exist "
                "or both be absent."
            )

    except ValidationError as error:
        raise CollectionValidationError(
            f"Invalid collection record: {error}"
        ) from error

    validate_collection(
        pdf_files=pdf_files,
        documents=documents,
        pages=pages,
        queries=queries,
        qrels=qrels,
    )

    return Collection(
        pdfs_path=pdfs_path,
        ragit_path=ragit_path,
        documents=documents,
        pages=pages,
        queries=queries,
        qrels=qrels,
    )


def create_corpus(
    pdf_files: list[Path],
    page_counts: dict[str, int],
) -> list[Page]:
    """Create deterministic page records for raw PDF collections."""
    pages: list[Page] = []
    page_id = 0

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem

        for page_number in range(page_counts[doc_id]):
            pages.append(
                Page(
                    page_id=page_id,
                    doc_id=doc_id,
                    page_number=page_number,
                )
            )
            page_id += 1

    return pages


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read JSON Lines records."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CollectionValidationError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise CollectionValidationError(
                    f"Expected a JSON object in {path} at line "
                    f"{line_number}."
                )

            records.append(record)

    return records


def read_qrels(
    path: Path,
) -> list[Qrel]:
    """Read canonical page-level qrels."""
    qrels: list[Qrel] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_columns = {
            "query_id",
            "page_id",
            "score",
        }

        if reader.fieldnames is None:
            raise CollectionValidationError(
                f"Qrels file has no header: {path}"
            )

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise CollectionValidationError(
                "Qrels file is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                qrels.append(
                    Qrel(
                        query_id=int(row["query_id"]),
                        page_id=int(row["page_id"]),
                        score=int(row["score"]),
                    )
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise CollectionValidationError(
                    f"Invalid qrel in {path} at line "
                    f"{line_number}: {error}"
                ) from error

    return qrels


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Write JSON Lines records atomically."""
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(path)


def copy_file(
    source: Path,
    destination: Path,
) -> None:
    """Copy a supplied file to its canonical collection location."""
    if not source.exists():
        raise CollectionInitializationError(
            f"Supplied file does not exist: {source}"
        )

    if not source.is_file():
        raise CollectionInitializationError(
            f"Supplied path is not a file: {source}"
        )

    if source.resolve() == destination.resolve():
        return

    shutil.copyfile(
        source,
        destination,
    )