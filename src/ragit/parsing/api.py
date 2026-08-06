"""Public high-level parsing API."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from ragit.data.models import Collection, Document, Page
from ragit.parsing.base import BaseParser
from ragit.parsing.models import (
    ParsedPage,
    ParsingConfig,
    ParsingResult,
    ParsingStorageError,
    ParsingValidationError,
)
from ragit.parsing.registry import get_parser
from ragit.parsing.storage import (
    configuration_hash,
    load_parsing_result,
    parsing_output_path,
    save_parsing_result,
)


def parse_collection(
    collection: Collection,
    config: ParsingConfig,
    overwrite: bool = False,
) -> ParsingResult:
    """Parse all PDF documents in an initialized collection.

    With ``overwrite=False``, all existing cached page records are reused,
    including successful, empty, and failed pages. Only missing page IDs are
    parsed.

    With ``overwrite=True``, the complete parsing artifact is rebuilt.
    """
    parser = _resolve_parser(config)
    pdfs_path = _resolve_pdfs_path(collection)

    documents_by_id = _index_documents(collection.documents)
    source_pages = _validate_and_sort_source_pages(collection.pages)

    source_pages_by_id = {
        page.page_id: page
        for page in source_pages
    }

    cached_pages_by_id: dict[int, ParsedPage]

    if overwrite:
        cached_pages_by_id = {}
    else:
        cached_pages_by_id = _load_cached_pages(
            pdfs_path=pdfs_path,
            config=config,
            source_pages_by_id=source_pages_by_id,
        )

    missing_pages = [
        page
        for page in source_pages
        if page.page_id not in cached_pages_by_id
    ]

    parsed_pages: list[ParsedPage] = []

    pages_by_document = _group_pages_by_document(
        missing_pages
    )

    for doc_id, document_pages in pages_by_document.items():
        document = documents_by_id.get(doc_id)

        if document is None:
            raise ParsingValidationError(
                f"Source pages reference unknown document ID "
                f"{doc_id!r}."
            )

        document_result = _parse_with_adapter(
            parser=parser,
            document=document,
            source_pages=document_pages,
            config=config,
        )

        parsed_pages.extend(document_result)

    combined_pages = _merge_expected_pages(
        source_pages=source_pages,
        cached_pages=cached_pages_by_id.values(),
        parsed_pages=parsed_pages,
    )

    artifact_path = parsing_output_path(
        pdfs_path=pdfs_path,
        config=config,
    )

    artifact_exists = artifact_path.exists()

    if (
        overwrite
        or missing_pages
        or not artifact_exists
    ):
        return save_parsing_result(
            pdfs_path=pdfs_path,
            config=config,
            pages=combined_pages,
            replace=artifact_exists,
        )

    return ParsingResult(
        config=config,
        config_hash=configuration_hash(config),
        output_path=artifact_path,
        pages=combined_pages,
    )


def parse_document(
    collection: Collection,
    doc_id: str,
    config: ParsingConfig,
    overwrite: bool = False,
) -> ParsingResult:
    """Parse one selected document from an initialized collection.

    The function uses the same collection-level parsing artifact as
    :func:`parse_collection`.

    With ``overwrite=False``, existing records for the selected document are
    reused and only missing page IDs are parsed.

    With ``overwrite=True``, all records for the selected document are
    replaced while records belonging to other documents are preserved.
    """
    parser = _resolve_parser(config)
    pdfs_path = _resolve_pdfs_path(collection)

    document = _find_document(
        collection=collection,
        doc_id=doc_id,
    )

    all_source_pages = _validate_and_sort_source_pages(
        collection.pages
    )

    all_source_pages_by_id = {
        page.page_id: page
        for page in all_source_pages
    }

    document_source_pages = sorted(
        [
            page
            for page in all_source_pages
            if page.doc_id == doc_id
        ],
        key=lambda page: page.page_number,
    )

    if not document_source_pages:
        raise ParsingValidationError(
            f"Document {doc_id!r} has no source pages."
        )

    cached_pages_by_id = _load_cached_pages(
        pdfs_path=pdfs_path,
        config=config,
        source_pages_by_id=all_source_pages_by_id,
        ignore_configuration_mismatch=overwrite,
    )

    if overwrite:
        preserved_pages_by_id = {
            page_id: page
            for page_id, page in cached_pages_by_id.items()
            if page.doc_id != doc_id
        }

        pages_to_parse = document_source_pages
    else:
        preserved_pages_by_id = dict(
            cached_pages_by_id
        )

        pages_to_parse = [
            page
            for page in document_source_pages
            if page.page_id not in cached_pages_by_id
        ]

    parsed_pages: list[ParsedPage] = []

    if pages_to_parse:
        parsed_pages = _parse_with_adapter(
            parser=parser,
            document=document,
            source_pages=pages_to_parse,
            config=config,
        )

    persisted_pages_by_id = dict(
        preserved_pages_by_id
    )

    for parsed_page in parsed_pages:
        persisted_pages_by_id[
            parsed_page.page_id
        ] = parsed_page

    persisted_pages = sorted(
        persisted_pages_by_id.values(),
        key=lambda page: page.page_id,
    )

    artifact_path = parsing_output_path(
        pdfs_path=pdfs_path,
        config=config,
    )

    artifact_exists = artifact_path.exists()

    if (
        overwrite
        or pages_to_parse
        or not artifact_exists
    ):
        stored_result = save_parsing_result(
            pdfs_path=pdfs_path,
            config=config,
            pages=persisted_pages,
            replace=artifact_exists,
        )
    else:
        stored_result = ParsingResult(
            config=config,
            config_hash=configuration_hash(config),
            output_path=artifact_path,
            pages=persisted_pages,
        )

    stored_pages_by_id = {
        page.page_id: page
        for page in stored_result.pages
    }

    selected_pages = [
        stored_pages_by_id[source_page.page_id]
        for source_page in document_source_pages
        if source_page.page_id
        in stored_pages_by_id
    ]

    return ParsingResult(
        config=stored_result.config,
        config_hash=stored_result.config_hash,
        output_path=stored_result.output_path,
        pages=selected_pages,
    )


def _resolve_parser(
    config: ParsingConfig,
) -> BaseParser:
    """Resolve, instantiate, and validate a parser adapter."""
    parser_class = get_parser(config.parser_name)
    parser = parser_class()

    parser.validate_config(config)

    return parser


def _resolve_pdfs_path(
    collection: Collection,
) -> Path:
    """Return the collection's PDF root directory."""
    pdfs_path = getattr(
        collection,
        "pdfs_path",
        None,
    )

    if pdfs_path is None:
        raise ParsingValidationError(
            "The initialized collection does not expose "
            "'pdfs_path'."
        )

    return Path(pdfs_path)


def _index_documents(
    documents: Iterable[Document],
) -> dict[str, Document]:
    """Index documents by their stable document IDs."""
    documents_by_id: dict[str, Document] = {}

    for document in documents:
        if document.doc_id in documents_by_id:
            raise ParsingValidationError(
                f"Duplicate document ID: "
                f"{document.doc_id!r}."
            )

        documents_by_id[
            document.doc_id
        ] = document

    return documents_by_id


def _find_document(
    collection: Collection,
    doc_id: str,
) -> Document:
    """Find one collection document by ID."""
    for document in collection.documents:
        if document.doc_id == doc_id:
            return document

    raise ParsingValidationError(
        f"Unknown document ID: {doc_id!r}."
    )


def _validate_and_sort_source_pages(
    source_pages: Iterable[Page],
) -> list[Page]:
    """Validate source-page uniqueness and sort by page ID."""
    pages_by_id: dict[int, Page] = {}
    document_page_numbers: set[
        tuple[str, int]
    ] = set()

    for page in source_pages:
        if page.page_id in pages_by_id:
            raise ParsingValidationError(
                f"Duplicate source page ID: "
                f"{page.page_id}."
            )

        document_page_key = (
            page.doc_id,
            page.page_number,
        )

        if (
            document_page_key
            in document_page_numbers
        ):
            raise ParsingValidationError(
                "Duplicate page number "
                f"{page.page_number} for document "
                f"{page.doc_id!r}."
            )

        pages_by_id[page.page_id] = page
        document_page_numbers.add(
            document_page_key
        )

    return sorted(
        pages_by_id.values(),
        key=lambda page: page.page_id,
    )


def _group_pages_by_document(
    source_pages: Iterable[Page],
) -> dict[str, list[Page]]:
    """Group source pages by document ID."""
    grouped: dict[str, list[Page]] = (
        defaultdict(list)
    )

    for page in source_pages:
        grouped[page.doc_id].append(page)

    return {
        doc_id: sorted(
            document_pages,
            key=lambda page: page.page_number,
        )
        for doc_id, document_pages in sorted(
            grouped.items()
        )
    }


def _parse_with_adapter(
    *,
    parser: BaseParser,
    document: Document,
    source_pages: Sequence[Page],
    config: ParsingConfig,
) -> list[ParsedPage]:
    """Invoke one low-level parser adapter and validate its output."""
    try:
        parsed_pages = parser.parse_document(
            document=document,
            source_pages=source_pages,
            config=config,
        )
    except Exception as error:
        # This is a final orchestration safeguard. Normally the adapter
        # already converts document-level errors into failed placeholders.
        parsed_pages = parser.create_failed_pages(
            source_pages=source_pages,
            config=config,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    _validate_parser_output(
        source_pages=source_pages,
        parsed_pages=parsed_pages,
    )

    return sorted(
        parsed_pages,
        key=lambda page: page.page_id,
    )


def _validate_parser_output(
    *,
    source_pages: Sequence[Page],
    parsed_pages: Sequence[ParsedPage],
) -> None:
    """Ensure an adapter returned exactly one record per expected page."""
    expected_pages_by_id = {
        page.page_id: page
        for page in source_pages
    }

    parsed_pages_by_id: dict[
        int,
        ParsedPage,
    ] = {}

    for parsed_page in parsed_pages:
        if (
            parsed_page.page_id
            in parsed_pages_by_id
        ):
            raise ParsingValidationError(
                "Parser returned duplicate page ID "
                f"{parsed_page.page_id}."
            )

        source_page = expected_pages_by_id.get(
            parsed_page.page_id
        )

        if source_page is None:
            raise ParsingValidationError(
                "Parser returned unexpected page ID "
                f"{parsed_page.page_id}."
            )

        if (
            parsed_page.doc_id
            != source_page.doc_id
        ):
            raise ParsingValidationError(
                "Parser returned an incorrect doc_id "
                f"for page {parsed_page.page_id}."
            )

        if (
            parsed_page.page_number
            != source_page.page_number
        ):
            raise ParsingValidationError(
                "Parser returned an incorrect "
                f"page_number for page "
                f"{parsed_page.page_id}."
            )

        parsed_pages_by_id[
            parsed_page.page_id
        ] = parsed_page

    missing_page_ids = (
        set(expected_pages_by_id)
        - set(parsed_pages_by_id)
    )

    if missing_page_ids:
        raise ParsingValidationError(
            "Parser did not return results for page IDs: "
            f"{sorted(missing_page_ids)}."
        )


def _load_cached_pages(
    *,
    pdfs_path: Path,
    config: ParsingConfig,
    source_pages_by_id: dict[int, Page],
    ignore_configuration_mismatch: bool = False,
) -> dict[int, ParsedPage]:
    """Load and validate an existing parsing artifact."""
    artifact_path = parsing_output_path(
        pdfs_path=pdfs_path,
        config=config,
    )

    if not artifact_path.exists():
        return {}

    result = load_parsing_result(
        pdfs_path=pdfs_path,
        parser_name=config.parser_name,
        config_hash=configuration_hash(config),
    )

    saved_config = _normalized_config(
        result.config
    )
    current_config = _normalized_config(
        config
    )

    if saved_config != current_config:
        if ignore_configuration_mismatch:
            return {}

        raise ParsingStorageError(
            "Parsing cache configuration mismatch. "
            "The saved config.json does not match "
            "the current normalized ParsingConfig."
        )

    cached_pages_by_id: dict[
        int,
        ParsedPage,
    ] = {}

    for cached_page in result.pages:
        if (
            cached_page.page_id
            in cached_pages_by_id
        ):
            raise ParsingStorageError(
                "Parsing cache contains duplicate "
                f"page ID {cached_page.page_id}."
            )

        source_page = source_pages_by_id.get(
            cached_page.page_id
        )

        if source_page is None:
            raise ParsingStorageError(
                f"Cached page ID "
                f"{cached_page.page_id} does not "
                "exist in the current collection."
            )

        if (
            cached_page.doc_id
            != source_page.doc_id
        ):
            raise ParsingStorageError(
                f"Cached page "
                f"{cached_page.page_id} has an "
                "unexpected doc_id."
            )

        if (
            cached_page.page_number
            != source_page.page_number
        ):
            raise ParsingStorageError(
                f"Cached page "
                f"{cached_page.page_id} has an "
                "unexpected page_number."
            )

        cached_pages_by_id[
            cached_page.page_id
        ] = cached_page

    return cached_pages_by_id


def _normalized_config(
    config: ParsingConfig,
) -> dict[str, object]:
    """Return a deterministic JSON-compatible config representation."""
    return config.model_dump(
        mode="json",
        exclude_none=False,
    )


def _merge_expected_pages(
    *,
    source_pages: Sequence[Page],
    cached_pages: Iterable[ParsedPage],
    parsed_pages: Iterable[ParsedPage],
) -> list[ParsedPage]:
    """Merge cache and newly parsed records into a complete result."""
    merged_pages_by_id: dict[
        int,
        ParsedPage,
    ] = {}

    for cached_page in cached_pages:
        if (
            cached_page.page_id
            in merged_pages_by_id
        ):
            raise ParsingValidationError(
                "Duplicate cached page ID "
                f"{cached_page.page_id}."
            )

        merged_pages_by_id[
            cached_page.page_id
        ] = cached_page

    for parsed_page in parsed_pages:
        merged_pages_by_id[
            parsed_page.page_id
        ] = parsed_page

    expected_page_ids = {
        page.page_id
        for page in source_pages
    }

    missing_page_ids = (
        expected_page_ids
        - set(merged_pages_by_id)
    )

    if missing_page_ids:
        raise ParsingValidationError(
            "No parsing output exists for page IDs: "
            f"{sorted(missing_page_ids)}."
        )

    unexpected_page_ids = (
        set(merged_pages_by_id)
        - expected_page_ids
    )

    if unexpected_page_ids:
        raise ParsingValidationError(
            "Parsing output contains unexpected page IDs: "
            f"{sorted(unexpected_page_ids)}."
        )

    return [
        merged_pages_by_id[source_page.page_id]
        for source_page in sorted(
            source_pages,
            key=lambda page: page.page_id,
        )
    ]