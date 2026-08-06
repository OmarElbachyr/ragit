"""Normalized models for parsed PDF pages."""
"""Normalized models for document parsing."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


ContentFormat = Literal["text", "markdown"]


class ParsingError(Exception):
    """Base exception for parsing-layer failures."""


class ParsingConfigurationError(ParsingError):
    """Raised when a parser configuration is invalid."""


class ParsingValidationError(ParsingError):
    """Raised when normalized parsing output is invalid."""


class ParsingStorageError(ParsingError):
    """Raised when parsing artifacts cannot be saved or loaded."""
    

class PageStatus(str, Enum):
    """Normalized state of a parsed page."""

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


class ParsingConfig(BaseModel):
    """Configuration for one parser execution."""

    model_config = ConfigDict(extra="forbid")

    parser_name: str
    output_format: ContentFormat
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parser_name")
    @classmethod
    def validate_parser_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("parser_name must not be empty.")

        if not all(
            character.isalnum() or character in {"_", "-"}
            for character in value
        ):
            raise ValueError(
                "parser_name may contain only letters, numbers, "
                "underscores, and hyphens."
            )

        return value


class ParsedPage(BaseModel):
    """Normalized output for one source PDF page."""

    model_config = ConfigDict(extra="forbid")

    page_id: StrictInt
    doc_id: str
    page_number: StrictInt
    content: str | None
    content_format: ContentFormat
    status: PageStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("doc_id must not be empty.")

        return value

    @field_validator("page_id", "page_number")
    @classmethod
    def validate_non_negative_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Page identifiers and numbers must be non-negative.")

        return value

    @model_validator(mode="after")
    def validate_status_content(self) -> "ParsedPage":
        if self.status == PageStatus.SUCCESS:
            if self.content is None or not self.content.strip():
                raise ValueError(
                    "A successful page must contain non-empty content."
                )

        elif self.status == PageStatus.EMPTY:
            if self.content != "":
                raise ValueError(
                    "An empty page must have content equal to an empty string."
                )

        elif self.status == PageStatus.FAILED:
            if self.content is not None:
                raise ValueError(
                    "A failed page must have content equal to None."
                )

        return self


class ParsingResult(BaseModel):
    """A loaded or newly persisted parsing result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ParsingConfig
    config_hash: str
    output_path: Path
    pages: list[ParsedPage]