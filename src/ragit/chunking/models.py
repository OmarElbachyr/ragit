"""Normalized models for RAGit chunking."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class ChunkingConfig(BaseModel):
    """Configuration for one chunking strategy."""

    model_config = ConfigDict(extra="forbid")

    chunker_name: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunker_name")
    @classmethod
    def validate_chunker_name(cls, value: str) -> str:
        """Require a non-empty chunker name."""
        normalized = value.strip()

        if not normalized:
            raise ValueError("chunker_name must not be empty.")

        return normalized


class Chunk(BaseModel):
    """One normalized retrieval chunk with page-level provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    page_ids: list[StrictInt]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "doc_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Require non-empty identifiers."""
        if not value:
            raise ValueError("Identifiers must not be empty.")

        return value

    @field_validator("page_ids")
    @classmethod
    def validate_page_ids(cls, value: list[int]) -> list[int]:
        """Require at least one unique source page ID."""
        if not value:
            raise ValueError("page_ids must contain at least one page ID.")

        if len(value) != len(set(value)):
            raise ValueError("page_ids must not contain duplicates.")

        return value


class ChunkingResult(BaseModel):
    """Persisted chunking result with provenance for downstream stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: list[Chunk]
    config_hash: str
    parsing_config_hash: str
    output_path: Path
