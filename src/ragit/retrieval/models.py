"""Normalized models for RAGit retrieval."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator


class RetrievalConfig(BaseModel):
    """Configuration for page-level retrieval in an experiment."""

    model_config = ConfigDict(extra="forbid")

    top_k: StrictInt = 100
    aggregation: Literal["max", "sum", "mean"] = "max"

    @field_validator("top_k")
    @classmethod
    def validate_config_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be greater than zero.")
        return value


class RetrievedChunk(BaseModel):
    """One ranked chunk-level retrieval result used during page aggregation."""

    model_config = ConfigDict(extra="forbid")

    query_id: StrictInt
    chunk_id: str
    page_ids: list[StrictInt]
    score: float
    rank: StrictInt

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value:
            raise ValueError("chunk_id must not be empty.")
        return value

    @field_validator("page_ids")
    @classmethod
    def validate_page_ids(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("page_ids must contain at least one page ID.")
        if len(value) != len(set(value)):
            raise ValueError("page_ids must not contain duplicates.")
        return value

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rank must be greater than zero.")
        return value


class RetrievedPage(BaseModel):
    """One final ranked page with all contributing chunk provenance."""

    model_config = ConfigDict(extra="forbid")

    query_id: StrictInt
    page_id: StrictInt
    score: float
    rank: StrictInt
    chunks: list[RetrievedChunk]

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rank must be greater than zero.")
        return value

    @field_validator("chunks")
    @classmethod
    def validate_chunks_non_empty(
        cls,
        value: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not value:
            raise ValueError("chunks must contain at least one contributor.")
        return value

    @model_validator(mode="after")
    def validate_chunk_provenance(self) -> "RetrievedPage":
        for chunk in self.chunks:
            if chunk.query_id != self.query_id:
                raise ValueError(
                    "All contributing chunks must belong to the same query."
                )
            if self.page_id not in chunk.page_ids:
                raise ValueError(
                    "Every contributing chunk must reference the retrieved page."
                )
        return self


class RetrievalResult(BaseModel):
    """One persisted page-level retrieval run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[RetrievedPage]
    config_hash: str
    output_path: Path
    top_k: StrictInt

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be greater than zero.")
        return value
