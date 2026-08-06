"""PyMuPDF4LLM parser adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
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


# Users may currently configure only this option.
_ALLOWED_OPTIONS = {
    "use_ocr",
}


# PyMuPDF4LLM uses Tesseract language codes.
_LANGUAGE_CODES = {
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "fr": "fra",
    "fra": "fra",
    "fre": "fra",
    "french": "fra",
    "de": "deu",
    "deu": "deu",
    "ger": "deu",
    "german": "deu",
    "lb": "ltz",
    "ltz": "ltz",
    "luxembourgish": "ltz",
    "es": "spa",
    "spa": "spa",
    "spanish": "spa",
    "ar": "ara",
    "ara": "ara",
    "arabic": "ara",
}


# Explicit RAGit defaults for Markdown extraction.
_MARKDOWN_DEFAULTS: dict[str, Any] = {
    "detect_bg_color": True,
    "dpi": 150,
    "embed_images": False,
    "extract_words": False,
    "filename": None,
    "fontsize_limit": 3,
    "footer": True,
    "force_text": True,
    "graphics_limit": None,
    "hdr_info": None,
    "header": True,
    "ignore_alpha": False,
    "ignore_code": False,
    "ignore_graphics": False,
    "ignore_images": False,
    "image_format": "png",
    "image_path": "",
    "image_size_limit": 0.05,
    "margins": 0,
    "ocr_dpi": 300,
    "ocr_function": None,
    "page_height": None,
    "page_separators": False,
    "page_width": 612,
    "show_progress": True,
    "table_strategy": "lines_strict",
    "use_glyphs": False,
    "write_images": False,
}


# Explicit RAGit defaults for plain-text extraction.
_TEXT_DEFAULTS: dict[str, Any] = {
    "footer": True,
    "force_text": True,
    "header": True,
    "ignore_code": False,
    "ocr_dpi": 400,
    "show_progress": True,
}


class PyMuPDF4LLMParser(BaseParser):
    """Parse complete PDF documents with PyMuPDF4LLM."""

    parser_name: ClassVar[str] = "pymupdf4llm"

    supported_output_formats: ClassVar[frozenset[ContentFormat]] = frozenset(
        {"text", "markdown"}
    )

    default_language: ClassVar[str] = "eng"

    @classmethod
    def validate_config(cls, config: ParsingConfig) -> None:
        """Validate the RAGit parser configuration."""
        super().validate_config(config)

        unsupported_options = set(config.options) - _ALLOWED_OPTIONS

        if unsupported_options:
            raise ParsingConfigurationError(
                f"Unsupported options for {cls.parser_name!r}: "
                f"{sorted(unsupported_options)}. "
                "Currently supported options: ['use_ocr']."
            )

        use_ocr = config.options.get("use_ocr", False)

        if not isinstance(use_ocr, bool):
            raise ParsingConfigurationError(
                "The 'use_ocr' option must be a boolean."
            )

    def parse_document(
        self,
        document: Document,
        source_pages: Sequence[Page],
        config: ParsingConfig,
    ) -> list[ParsedPage]:
        """Parse one PDF into one normalized result per corpus page."""
        self.validate_config(config)
        self._validate_source_pages(document, source_pages)

        effective_options = self._build_effective_options(
            document=document,
            config=config,
        )

        try:
            backend = _load_backend()
            parser_version = _get_parser_version(backend)
        except Exception as error:
            return self.create_failed_pages(
                source_pages=source_pages,
                config=config,
                error_type=type(error).__name__,
                error_message=str(error),
                metadata={
                    "parser_version": None,
                    "parser_options": effective_options,
                },
            )

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

        parsed_pages: list[ParsedPage] = []

        for source_page in sorted(
            source_pages,
            key=lambda page: page.page_number,
        ):
            parsed_pages.append(
                self._parse_page(
                    backend=backend,
                    pdf_path=pdf_path,
                    source_page=source_page,
                    config=config,
                    parser_version=parser_version,
                    effective_options=effective_options,
                )
            )

        return parsed_pages

    @classmethod
    def _build_effective_options(
        cls,
        *,
        document: Document,
        config: ParsingConfig,
    ) -> dict[str, Any]:
        """Build the complete explicit backend configuration."""
        use_ocr = config.options.get("use_ocr", False)

        language = cls._resolve_document_language(document)

        if config.output_format == "markdown":
            defaults = _MARKDOWN_DEFAULTS
        else:
            defaults = _TEXT_DEFAULTS

        return {
            **defaults,
            # RAGit convention:
            # use_ocr=True means OCR every page.
            "use_ocr": use_ocr,
            "force_ocr": use_ocr,
            "ocr_language": language,
        }

    @classmethod
    def _resolve_document_language(
        cls,
        document: Document,
    ) -> str:
        """Resolve and normalize the document OCR language.

        Supported document metadata fields, in priority order:

        1. language
        2. doc_language
        3. langauge, for compatibility with the misspelled field

        Missing language metadata falls back to English.
        """
        raw_language: str | None = None

        for field_name in (
            "language",
            "doc_language",
            "langauge",
        ):
            value = getattr(document, field_name, None)

            if isinstance(value, str) and value.strip():
                raw_language = value.strip()
                break

        if raw_language is None:
            return cls.default_language

        normalized = raw_language.lower()

        return _LANGUAGE_CODES.get(normalized, normalized)

    def _parse_page(
        self,
        *,
        backend: ModuleType,
        pdf_path: Path,
        source_page: Page,
        config: ParsingConfig,
        parser_version: str | None,
        effective_options: dict[str, Any],
    ) -> ParsedPage:
        """Parse one expected source page."""
        metadata = {
            "parser_name": self.parser_name,
            "parser_version": parser_version,
            "parser_options": effective_options,
        }

        try:
            chunks = self._extract_page_chunks(
                backend=backend,
                pdf_path=pdf_path,
                page_number=source_page.page_number,
                output_format=config.output_format,
                effective_options=effective_options,
            )

            if not chunks:
                return self._failed_page(
                    source_page=source_page,
                    config=config,
                    metadata=metadata,
                    error_type="MissingPageOutput",
                    error_message=(
                        "PyMuPDF4LLM returned no output for zero-based "
                        f"page {source_page.page_number}."
                    ),
                )

            chunk = chunks[0]

            if not isinstance(chunk, dict):
                return self._failed_page(
                    source_page=source_page,
                    config=config,
                    metadata=metadata,
                    error_type="InvalidPageOutput",
                    error_message=(
                        "PyMuPDF4LLM returned a non-dictionary page output."
                    ),
                )

            content = chunk.get("text")

            if not isinstance(content, str):
                return self._failed_page(
                    source_page=source_page,
                    config=config,
                    metadata=metadata,
                    error_type="InvalidPageOutput",
                    error_message=(
                        "PyMuPDF4LLM output does not contain string content."
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
    def _extract_page_chunks(
        *,
        backend: ModuleType,
        pdf_path: Path,
        page_number: int,
        output_format: ContentFormat,
        effective_options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Call the native PyMuPDF4LLM output method."""
        options = {
            **effective_options,
            "pages": [page_number],
            "page_chunks": True,
        }

        if output_format == "markdown":
            result = backend.to_markdown(
                str(pdf_path),
                **options,
            )
        else:
            result = backend.to_text(
                str(pdf_path),
                **options,
            )

        if not isinstance(result, list):
            raise TypeError(
                "PyMuPDF4LLM must return a list when page_chunks=True."
            )

        return result

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
        """Validate that source pages belong to the requested document."""
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


def _load_backend() -> ModuleType:
    """Import PyMuPDF4LLM only when the adapter is used."""
    try:
        import pymupdf4llm
    except ImportError as error:
        raise ImportError(
            "PyMuPDF4LLM is required for the 'pymupdf4llm' parser. "
            "Install it with: pip install pymupdf4llm"
        ) from error

    return pymupdf4llm


def _get_parser_version(
    backend: ModuleType,
) -> str | None:
    """Read the installed PyMuPDF4LLM version."""
    version = getattr(backend, "version", None)

    if callable(version):
        version = version()

    if version is None:
        version = getattr(backend, "__version__", None)

    return str(version) if version is not None else None


register_parser(PyMuPDF4LLMParser)