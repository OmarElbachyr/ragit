"""Validation for canonical ragit collections."""

from collections import Counter, defaultdict
from pathlib import Path

import fitz

from ragit.data.models import Document, Page, Qrel, Query


class RagitCollectionError(Exception):
    """Base exception for collection-related failures."""


class CollectionInitializationError(RagitCollectionError):
    """Raised when a collection cannot be initialized."""


class CollectionValidationError(RagitCollectionError):
    """Raised when canonical collection data is invalid."""


def validate_pdf_directory(pdfs_path: Path) -> list[Path]:
    """Validate a PDF directory and return PDFs in deterministic order."""
    if not pdfs_path.exists():
        raise CollectionInitializationError(
            f"PDF directory does not exist: {pdfs_path}"
        )

    if not pdfs_path.is_dir():
        raise CollectionInitializationError(
            f"PDF path is not a directory: {pdfs_path}"
        )

    pdf_files = sorted(
        (
            path
            for path in pdfs_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: (path.name.lower(), path.name),
    )

    if not pdf_files:
        raise CollectionInitializationError(
            f"No PDF files found in: {pdfs_path}"
        )

    doc_ids = [path.stem for path in pdf_files]

    duplicates = sorted(
        doc_id
        for doc_id, count in Counter(doc_ids).items()
        if count > 1
    )

    if duplicates:
        raise CollectionValidationError(
            "Duplicate document IDs generated from PDF filenames: "
            + ", ".join(duplicates)
        )

    return pdf_files


def read_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    try:
        with fitz.open(pdf_path) as document:
            return document.page_count
    except Exception as error:
        raise CollectionInitializationError(
            f"Could not read PDF {pdf_path}: {error}"
        ) from error


def validate_collection(
    *,
    pdf_files: list[Path],
    documents: list[Document],
    pages: list[Page],
    queries: list[Query],
    qrels: list[Qrel],
) -> None:
    """Validate collection schemas and cross-file references."""
    _validate_documents_and_pages(
        pdf_files=pdf_files,
        documents=documents,
        pages=pages,
    )
    _validate_queries(queries)
    _validate_qrels(
        pages=pages,
        queries=queries,
        qrels=qrels,
    )


def _validate_documents_and_pages(
    *,
    pdf_files: list[Path],
    documents: list[Document],
    pages: list[Page],
) -> None:
    pdf_by_name = {pdf.name: pdf for pdf in pdf_files}
    actual_page_counts = {
        pdf.stem: read_pdf_page_count(pdf)
        for pdf in pdf_files
    }

    document_ids = [document.doc_id for document in documents]
    file_names = [document.file_name for document in documents]

    _require_unique(document_ids, "document ID")
    _require_unique(file_names, "document filename")

    expected_doc_ids = set(actual_page_counts)
    metadata_doc_ids = set(document_ids)

    if metadata_doc_ids != expected_doc_ids:
        missing = sorted(expected_doc_ids - metadata_doc_ids)
        unknown = sorted(metadata_doc_ids - expected_doc_ids)

        details = []

        if missing:
            details.append(f"missing metadata for: {missing}")

        if unknown:
            details.append(f"metadata references unknown documents: {unknown}")

        raise CollectionValidationError(
            "Document metadata does not match the PDF directory: "
            + "; ".join(details)
        )

    for document in documents:
        if document.file_name not in pdf_by_name:
            raise CollectionValidationError(
                f"Metadata references missing PDF: {document.file_name}"
            )

        expected_doc_id = Path(document.file_name).stem

        if document.doc_id != expected_doc_id:
            raise CollectionValidationError(
                f"Document {document.file_name!r} has doc_id "
                f"{document.doc_id!r}; expected {expected_doc_id!r}."
            )

        actual_count = actual_page_counts[document.doc_id]

        if document.page_number != actual_count:
            raise CollectionValidationError(
                f"Metadata page count for {document.doc_id!r} is "
                f"{document.page_number}; PDF contains {actual_count} pages."
            )

    page_ids = [page.page_id for page in pages]
    _require_contiguous_ids(page_ids, "page")

    pages_by_document: dict[str, list[int]] = defaultdict(list)

    for page in pages:
        if page.doc_id not in expected_doc_ids:
            raise CollectionValidationError(
                f"Corpus page {page.page_id} references unknown document "
                f"{page.doc_id!r}."
            )

        if page.page_number < 0:
            raise CollectionValidationError(
                f"Page {page.page_id} has negative page_number "
                f"{page.page_number}."
            )

        actual_count = actual_page_counts[page.doc_id]

        if page.page_number >= actual_count:
            raise CollectionValidationError(
                f"Page {page.page_id} references page_number "
                f"{page.page_number} in {page.doc_id!r}, but the PDF "
                f"contains {actual_count} pages."
            )

        pages_by_document[page.doc_id].append(page.page_number)

    for doc_id, actual_count in actual_page_counts.items():
        page_numbers = sorted(pages_by_document.get(doc_id, []))
        expected_numbers = list(range(actual_count))

        if page_numbers != expected_numbers:
            raise CollectionValidationError(
                f"Corpus pages for {doc_id!r} must be exactly "
                f"0 through {actual_count - 1}."
            )


def _validate_queries(queries: list[Query]) -> None:
    query_ids = [query.query_id for query in queries]

    if query_ids:
        _require_contiguous_ids(query_ids, "query")

    for query in queries:
        if not query.text.strip():
            raise CollectionValidationError(
                f"Query {query.query_id} has empty text."
            )


def _validate_qrels(
    *,
    pages: list[Page],
    queries: list[Query],
    qrels: list[Qrel],
) -> None:
    if bool(queries) != bool(qrels):
        raise CollectionValidationError(
            "Queries and qrels must both be present or both be absent."
        )

    page_ids = {page.page_id for page in pages}
    query_ids = {query.query_id for query in queries}
    seen_pairs: set[tuple[int, int]] = set()

    for qrel in qrels:
        if qrel.query_id not in query_ids:
            raise CollectionValidationError(
                f"Qrel references unknown query ID: {qrel.query_id}"
            )

        if qrel.page_id not in page_ids:
            raise CollectionValidationError(
                f"Qrel references unknown page ID: {qrel.page_id}"
            )

        pair = (qrel.query_id, qrel.page_id)

        if pair in seen_pairs:
            raise CollectionValidationError(
                f"Duplicate qrel for query {qrel.query_id} "
                f"and page {qrel.page_id}."
            )

        seen_pairs.add(pair)


def _require_unique(values: list[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise CollectionValidationError(
            f"{label.capitalize()} values must be unique."
        )


def _require_contiguous_ids(ids: list[int], label: str) -> None:
    _require_unique(ids, f"{label} ID")

    expected = list(range(len(ids)))

    if sorted(ids) != expected:
        raise CollectionValidationError(
            f"{label.capitalize()} IDs must start at 0 and be contiguous. "
            f"Expected {expected[:5]}"
            f"{'...' if len(expected) > 5 else ''}, "
            f"received {sorted(ids)[:5]}"
            f"{'...' if len(ids) > 5 else ''}."
        )