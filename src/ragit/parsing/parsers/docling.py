"""Docling parser adapter."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar

from ragit.data.models import Document, Page
from ragit.parsing.base import BaseParser
from ragit.parsing.models import (
    ContentFormat,
    PageStatus,
    ParsedPage,
    ParsingConfig,
    ParsingConfigurationError,
)
from ragit.parsing.registry import register_parser


# Users currently configure only whether OCR is enabled.
_ALLOWED_OPTIONS = {
    "use_ocr",
}


# EasyOCR uses ISO 639-1-style language codes.
_LANGUAGE_CODES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "lb": "lb",
    "ltz": "lb",
    "luxembourgish": "lb",
}


# Explicit RAGit defaults for Docling.
_DOCLING_DEFAULTS: dict[str, Any] = {
    "do_ocr": False,
    "ocr_mode": "default",
    "ocr_language": ["en"],
    "do_table_structure": True,
    "table_mode": "accurate",
    "do_cell_matching": True,
    "do_formula_enrichment": False,
    "do_code_enrichment": False,
    "do_picture_description": False,
    "do_picture_classification": False,
}


class DoclingParser(BaseParser):
    """Parse complete PDF documents using Docling."""

    parser_name: ClassVar[str] = "docling"

    supported_output_formats: ClassVar[frozenset[ContentFormat]] = frozenset(
        {"text", "markdown"}
    )

    supports_language: ClassVar[bool] = True
    default_language: ClassVar[str] = "en"

    @classmethod
    def validate_config(cls, config: ParsingConfig) -> None:
        """Validate output format and user options."""
        super().validate_config(config)

        unsupported_options = set(config.options) - _ALLOWED_OPTIONS

        if unsupported_options:
            raise ParsingConfigurationError(
                f"Unsupported options for {cls.parser_name!r}: "
                f"{sorted(unsupported_options)}. "
                f"Supported options: {sorted(_ALLOWED_OPTIONS)}."
            )

        use_ocr = config.options.get("use_ocr", False)

        if not isinstance(use_ocr, bool):
            raise ParsingConfigurationError(
                "The 'use_ocr' option must be a boolean."
            )

    @classmethod
    def _normalize_language(
        cls,
        language: str | None,
    ) -> str:
        """Convert document metadata language to an EasyOCR code."""
        if not language:
            return cls.default_language

        normalized = language.strip().lower()

        return _LANGUAGE_CODES.get(normalized, normalized)

    @classmethod
    def _effective_options(
        cls,
        document: Document,
        config: ParsingConfig,
    ) -> dict[str, Any]:
        """Build the complete explicit Docling configuration."""
        use_ocr = config.options.get("use_ocr", False)

        document_language = cls.resolve_document_language(document)
        ocr_language = cls._normalize_language(document_language)

        return {
            **_DOCLING_DEFAULTS,
            "do_ocr": use_ocr,
            # RAGit convention: use_ocr=True means OCR every page.
            "ocr_mode": "full_page" if use_ocr else "default",
            "ocr_language": [ocr_language],
        }

    def parse_document(
        self,
        document: Document,
        source_pages: Sequence[Page],
        config: ParsingConfig,
    ) -> list[ParsedPage]:
        """Parse one PDF and return one record per corpus page."""
        self.validate_config(config)
        self._validate_source_pages(document, source_pages)

        effective_options = self._effective_options(
            document=document,
            config=config,
        )

        parser_version = _get_docling_version()
        pdf_path = Path(document.local_path)

        if not pdf_path.exists():
            return self.create_failed_pages(
                source_pages=source_pages,
                config=config,
                error_type="FileNotFoundError",
                error_message=f"PDF file does not exist: {pdf_path}",
                metadata={
                    "parser_version": parser_version,
                    "parser_options": effective_options,
                },
            )

        try:
            converter = _build_converter(effective_options)
            conversion_result = converter.convert(str(pdf_path))
            docling_document = conversion_result.document

            if docling_document is None:
                raise RuntimeError(
                    "Docling conversion returned no document."
                )

        except Exception as error:
            return self.create_failed_pages(
                source_pages=source_pages,
                config=config,
                error_type=type(error).__name__,
                error_message=str(error),
                metadata={
                    "parser_version": parser_version,
                    "parser_options": effective_options,
                },
            )

        parsed_pages: list[ParsedPage] = []

        for source_page in sorted(
            source_pages,
            key=lambda page: page.page_number,
        ):
            parsed_pages.append(
                self._export_page(
                    docling_document=docling_document,
                    source_page=source_page,
                    config=config,
                    parser_version=parser_version,
                    effective_options=effective_options,
                )
            )

        return parsed_pages

    def _export_page(
        self,
        *,
        docling_document: Any,
        source_page: Page,
        config: ParsingConfig,
        parser_version: str | None,
        effective_options: dict[str, Any],
    ) -> ParsedPage:
        """Export one normalized page from the converted document."""
        metadata = {
            "parser_name": self.parser_name,
            "parser_version": parser_version,
            "parser_options": effective_options,
        }

        # Docling uses one-based page numbers.
        docling_page_number = source_page.page_number + 1

        try:
            content = self._native_page_export(
                docling_document=docling_document,
                docling_page_number=docling_page_number,
                output_format=config.output_format,
                use_ocr=effective_options["do_ocr"],
            )

            if not isinstance(content, str):
                return self._failed_page(
                    source_page=source_page,
                    config=config,
                    metadata=metadata,
                    error_type="MissingPageOutput",
                    error_message=(
                        "Docling returned no string output for "
                        f"one-based page {docling_page_number}."
                    ),
                )

            if not content.strip():
                return ParsedPage(
                    page_id=source_page.page_id,
                    doc_id=source_page.doc_id,
                    page_number=source_page.page_number,
                    content="",
                    content_format=config.output_format,
                    status=PageStatus.EMPTY,
                    metadata=metadata,
                )

            return ParsedPage(
                page_id=source_page.page_id,
                doc_id=source_page.doc_id,
                page_number=source_page.page_number,
                content=content,
                content_format=config.output_format,
                status=PageStatus.SUCCESS,
                metadata=metadata,
            )

        except Exception as error:
            return self._failed_page(
                source_page=source_page,
                config=config,
                metadata=metadata,
                error_type=type(error).__name__,
                error_message=str(error),
            )

    @staticmethod
    def _native_page_export(
        *,
        docling_document: Any,
        docling_page_number: int,
        output_format: ContentFormat,
        use_ocr: bool,
    ) -> str:
        """Use Docling's native page export method."""
        if output_format == "markdown":
            return docling_document.export_to_markdown(
                page_no=docling_page_number,
                traverse_pictures=use_ocr,
            )

        return docling_document.export_to_text(
            page_no=docling_page_number,
            traverse_pictures=use_ocr,
        )

    @staticmethod
    def _failed_page(
        *,
        source_page: Page,
        config: ParsingConfig,
        metadata: dict[str, Any],
        error_type: str,
        error_message: str,
    ) -> ParsedPage:
        """Create one normalized failed-page record."""
        return ParsedPage(
            page_id=source_page.page_id,
            doc_id=source_page.doc_id,
            page_number=source_page.page_number,
            content=None,
            content_format=config.output_format,
            status=PageStatus.FAILED,
            metadata={
                **metadata,
                "error_type": error_type,
                "error_message": error_message,
            },
        )

    @staticmethod
    def _validate_source_pages(
        document: Document,
        source_pages: Sequence[Page],
    ) -> None:
        """Validate source pages for one document."""
        seen_page_ids: set[int] = set()
        seen_page_numbers: set[int] = set()

        for page in source_pages:
            if page.doc_id != document.doc_id:
                raise ParsingConfigurationError(
                    f"Page {page.page_id} belongs to document "
                    f"{page.doc_id!r}, not {document.doc_id!r}."
                )

            if page.page_id in seen_page_ids:
                raise ParsingConfigurationError(
                    f"Duplicate source page_id: {page.page_id}"
                )

            if page.page_number in seen_page_numbers:
                raise ParsingConfigurationError(
                    f"Duplicate page_number for {document.doc_id!r}: "
                    f"{page.page_number}"
                )

            seen_page_ids.add(page.page_id)
            seen_page_numbers.add(page.page_number)


def _build_converter(
    effective_options: dict[str, Any],
) -> Any:
    """Create a Docling converter from explicit RAGit options."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            OcrMode,
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )
    except ImportError as error:
        raise ImportError(
            "Docling is required for the 'docling' parser. "
            "Install the RAGit Docling parser dependencies."
        ) from error

    ocr_mode = (
        OcrMode.FULL_PAGE
        if effective_options["do_ocr"]
        else OcrMode.DEFAULT
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=effective_options["do_ocr"],
        do_table_structure=effective_options[
            "do_table_structure"
        ],
        do_formula_enrichment=effective_options[
            "do_formula_enrichment"
        ],
        do_code_enrichment=effective_options[
            "do_code_enrichment"
        ],
        do_picture_description=effective_options[
            "do_picture_description"
        ],
        do_picture_classification=effective_options[
            "do_picture_classification"
        ],
    )

    pipeline_options.ocr_options = EasyOcrOptions(
        mode=ocr_mode,
        lang=effective_options["ocr_language"],
    )

    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )

    pipeline_options.table_structure_options.do_cell_matching = (
        effective_options["do_cell_matching"]
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def _get_docling_version() -> str | None:
    """Return the installed Docling package version."""
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


register_parser(DoclingParser)