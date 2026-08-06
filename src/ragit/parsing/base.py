"""Abstract parser contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from ragit.data.models import Document, Page
from ragit.parsing.models import (
    ContentFormat,
    PageStatus,
    ParsedPage,
    ParsingConfig,
    ParsingConfigurationError,
)


class BaseParser(ABC):
    """Base contract implemented by every parser adapter.

    A parser receives one complete PDF document and its known source pages.
    It must return exactly one normalized ParsedPage per source page.
    """

    parser_name: ClassVar[str]
    supported_output_formats: ClassVar[frozenset[ContentFormat]]

    # Override these only in parsers that support a language parameter.
    supports_language: ClassVar[bool] = False
    default_language: ClassVar[str | None] = None

    # "langauge" is temporarily supported for backward compatibility
    # with incorrectly named metadata fields.
    language_metadata_fields: ClassVar[tuple[str, ...]] = (
        "language",
        "doc_language",
        "langauge",
    )

    @classmethod
    def resolve_document_language(
        cls,
        document: Document,
    ) -> str | None:
        """Resolve the parser language from document metadata."""
        if not cls.supports_language:
            return None

        for field_name in cls.language_metadata_fields:
            value = getattr(document, field_name, None)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return cls.default_language

    @classmethod
    def validate_config(cls, config: ParsingConfig) -> None:
        """Validate a configuration against this parser."""
        if config.parser_name != cls.parser_name:
            raise ParsingConfigurationError(
                f"Configuration targets parser {config.parser_name!r}, "
                f"but this parser is {cls.parser_name!r}."
            )

        if config.output_format not in cls.supported_output_formats:
            supported = ", ".join(sorted(cls.supported_output_formats))

            raise ParsingConfigurationError(
                f"Parser {cls.parser_name!r} does not support output format "
                f"{config.output_format!r}. Supported formats: {supported}."
            )

    @abstractmethod
    def parse_document(
        self,
        document: Document,
        source_pages: Sequence[Page],
        config: ParsingConfig,
    ) -> list[ParsedPage]:
        """Parse one complete PDF into normalized page-level outputs."""

    @classmethod
    def create_failed_pages(
        cls,
        source_pages: Sequence[Page],
        config: ParsingConfig,
        *,
        error_type: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ParsedPage]:
        """Create failed placeholders after a document-level failure."""
        cls.validate_config(config)

        shared_metadata = {
            "parser_name": cls.parser_name,
            "error_type": error_type,
            "error_message": error_message,
            **(metadata or {}),
        }

        return [
            ParsedPage(
                page_id=page.page_id,
                doc_id=page.doc_id,
                page_number=page.page_number,
                content=None,
                content_format=config.output_format,
                status=PageStatus.FAILED,
                metadata=shared_metadata.copy(),
            )
            for page in source_pages
        ]