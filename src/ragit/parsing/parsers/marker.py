"""Marker parser adapter."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
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

_PAGE_SEPARATOR = "RAGIT-MARKER-PAGE-BREAK-3A3B7B53"
_MARKER_LLM_SERVICE = "marker.services.openai.OpenAIService"
_DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_VLLM_BASE_URL_ENV = "RAGIT_MARKER_VLLM_BASE_URL"
_VLLM_MODEL_ENV = "RAGIT_MARKER_VLLM_MODEL"
_PAGE_BOUNDARY_PATTERN = re.compile(
    rf"(?m)^\{{(?P<page_number>\d+)\}}{_PAGE_SEPARATOR}$"
)


class MarkerParser(BaseParser):
    """Parse complete PDF documents with Marker into Markdown."""

    parser_name: ClassVar[str] = "marker"

    supported_output_formats: ClassVar[frozenset[ContentFormat]] = frozenset(
        {"markdown"}
    )

    supported_options: ClassVar[frozenset[str]] = frozenset(
        {"use_ocr", "describe_pictures"}
    )

    @classmethod
    def validate_config(cls, config: ParsingConfig) -> None:
        """Validate Marker-specific configuration."""
        super().validate_config(config)

        use_ocr = config.options.get("use_ocr", False)
        describe_pictures = config.options.get(
            "describe_pictures",
            False,
        )

        if not isinstance(use_ocr, bool):
            raise ParsingConfigurationError(
                "Option 'use_ocr' must be a Boolean."
            )

        if not isinstance(describe_pictures, bool):
            raise ParsingConfigurationError(
                "Option 'describe_pictures' must be a Boolean."
            )

    @classmethod
    def _effective_options(
        cls,
        source_pages: Sequence[Page],
        config: ParsingConfig,
    ) -> dict[str, Any]:
        """Build the complete Marker configuration."""
        describe_pictures = config.options.get(
            "describe_pictures",
            False,
        )

        options = {
            "output_format": "markdown",
            "paginate_output": True,
            "page_separator": _PAGE_SEPARATOR,
            "page_range": sorted(
                page.page_number for page in source_pages
            ),
            # RAGit's common OCR control maps to Marker's full-document OCR.
            "force_ocr": config.options.get("use_ocr", False),
            # Marker describes pictures through its LLM path when extracted
            # image files are disabled.
            "use_llm": describe_pictures,
            "extract_images": False,
        }

        if describe_pictures:
            options.update(
                {
                    "openai_base_url": os.getenv(
                        _VLLM_BASE_URL_ENV,
                        _DEFAULT_VLLM_BASE_URL,
                    ),
                    "openai_model": os.getenv(
                        _VLLM_MODEL_ENV,
                        _DEFAULT_VLLM_MODEL,
                    ),
                    "openai_image_format": "png",
                    # vLLM accepts any non-empty key unless authentication is
                    # explicitly enabled on the local server.
                    "openai_api_key": "EMPTY",
                }
            )

        return options

    def parse_document(
        self,
        document: Document,
        source_pages: Sequence[Page],
        config: ParsingConfig,
    ) -> list[ParsedPage]:
        """Parse one PDF and return one normalized page per corpus page."""
        self.validate_config(config)
        self._validate_source_pages(document, source_pages)

        effective_options = self._effective_options(source_pages, config)
        parser_version = _get_marker_version()
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
            backend = _load_backend()
            converter_options = {
                "artifact_dict": backend.create_model_dict(),
                "config": effective_options,
            }

            if config.options.get("describe_pictures", False):
                converter_options["llm_service"] = _MARKER_LLM_SERVICE

            converter = backend.PdfConverter(
                **converter_options,
            )
            rendered = converter(str(pdf_path))
            markdown, _, _ = backend.text_from_rendered(rendered)

            if not isinstance(markdown, str):
                raise TypeError(
                    "Marker returned non-string Markdown output."
                )

            markdown_by_page = _split_paginated_markdown(markdown)

        # Backend failures are normalized into RAGit's failed-page records.
        except Exception as error:  # noqa: BLE001
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

        return [
            self._normalize_page(
                source_page=source_page,
                content=markdown_by_page.get(source_page.page_number),
                config=config,
                parser_version=parser_version,
                effective_options=effective_options,
            )
            for source_page in sorted(
                source_pages,
                key=lambda page: page.page_number,
            )
        ]

    def _normalize_page(
        self,
        *,
        source_page: Page,
        content: str | None,
        config: ParsingConfig,
        parser_version: str | None,
        effective_options: dict[str, Any],
    ) -> ParsedPage:
        """Convert one paginated Marker section into a normalized page."""
        metadata = {
            "parser_name": self.parser_name,
            "parser_version": parser_version,
            "parser_options": effective_options,
        }

        if content is None:
            return self._failed_page(
                source_page=source_page,
                config=config,
                metadata=metadata,
                error_type="MissingPageOutput",
                error_message=(
                    "Marker returned no paginated Markdown output for "
                    f"zero-based page {source_page.page_number}."
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
        """Validate that all source pages belong to the document."""
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


def _split_paginated_markdown(markdown: str) -> dict[int, str]:
    """Split Marker paginated Markdown into zero-based page content."""
    boundaries = list(_PAGE_BOUNDARY_PATTERN.finditer(markdown))

    if not boundaries:
        raise ValueError(
            "Marker output did not contain the requested page boundaries."
        )

    output: dict[int, str] = {}

    for index, boundary in enumerate(boundaries):
        page_number = int(boundary.group("page_number"))
        content_start = boundary.end()
        content_end = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(markdown)
        )

        if page_number in output:
            raise ValueError(
                "Marker output included a duplicate page boundary for "
                f"zero-based page {page_number}."
            )

        output[page_number] = markdown[content_start:content_end].strip()

    return output


def _load_backend() -> ModuleType:
    """Import Marker only when this adapter is used."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as error:
        raise ImportError(
            "Marker is required for the 'marker' parser. Install it with: "
            "pip install -e '.[marker]'"
        ) from error

    backend = ModuleType("marker_backend")
    backend.PdfConverter = PdfConverter
    backend.create_model_dict = create_model_dict
    backend.text_from_rendered = text_from_rendered
    return backend


def _get_marker_version() -> str | None:
    """Return the installed Marker package version."""
    try:
        return version("marker-pdf")
    except PackageNotFoundError:
        return None


register_parser(MarkerParser)
