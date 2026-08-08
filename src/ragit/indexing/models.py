"""Normalized models for RAGit indexing."""

from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from ragit.chunking.models import Chunk


class IndexingConfig(BaseModel):
    """Configuration for one indexing run."""

    model_config = ConfigDict(extra="forbid")

    index_type: Literal["dense", "late_interaction"] = "dense"
    model_name: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model_name must not be empty.")
        return normalized


class IndexedChunk(BaseModel):
    """Mapping between a FAISS insertion position and a RAGit chunk ID."""

    model_config = ConfigDict(extra="forbid")

    index_position: StrictInt
    chunk_id: str

    @field_validator("index_position")
    @classmethod
    def validate_index_position(cls, value: int) -> int:
        if value < 0:
            raise ValueError("index_position must be non-negative.")
        return value

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value:
            raise ValueError("chunk_id must not be empty.")
        return value


class DenseIndexResult(BaseModel):
    """Loaded or newly built dense index and its source provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    chunk_mapping: list[IndexedChunk]
    config_hash: str
    output_path: Path
    config: IndexingConfig | None = None
    chunks: list[Chunk] = Field(default_factory=list)


class LateInteractionIndexResult(BaseModel):
    """Loaded or newly built PyLate index and its source provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    chunk_ids: list[str]
    config_hash: str
    output_path: Path
    config: IndexingConfig | None = None
    chunks: list[Chunk] = Field(default_factory=list)

    @field_validator("chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("chunk_ids must not contain duplicates.")
        if any(not chunk_id for chunk_id in value):
            raise ValueError("chunk_ids must not contain empty IDs.")
        return value


IndexResult: TypeAlias = DenseIndexResult | LateInteractionIndexResult
