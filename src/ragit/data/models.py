"""Data models for canonical RAGit collections."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt


class Page(BaseModel):
    """One PDF page in the canonical corpus."""

    model_config = ConfigDict(extra="forbid")

    page_id: StrictInt
    doc_id: str
    page_number: StrictInt


class Document(BaseModel):
    """Metadata for one PDF document."""

    model_config = ConfigDict(extra="allow")

    file_name: str
    doc_id: str
    local_path: str
    page_number: StrictInt

    def to_record(self) -> dict[str, Any]:
        """Return the complete record, including additional metadata fields."""
        return self.model_dump()


class Query(BaseModel):
    """One retrieval query."""

    model_config = ConfigDict(extra="allow")

    query_id: StrictInt
    text: str

    def to_record(self) -> dict[str, Any]:
        """Return the complete record, including optional fields."""
        return self.model_dump()


class Qrel(BaseModel):
    """One page-level relevance judgment."""

    model_config = ConfigDict(extra="forbid")

    query_id: StrictInt
    page_id: StrictInt
    score: StrictInt


class Collection(BaseModel):
    """A validated canonical RAGit collection."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    pdfs_path: Path
    ragit_path: Path
    documents: list[Document]
    pages: list[Page]
    queries: list[Query]
    qrels: list[Qrel]