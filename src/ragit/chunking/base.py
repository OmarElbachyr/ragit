"""Common document-level chunking contract and provenance mapping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragit.parsing.models import PageStatus, ParsedPage

from ragit.chunking.models import Chunk, ChunkingConfig


_PAGE_SEPARATOR = "\n"


@dataclass(frozen=True)
class _PageRange:
    """Character range occupied by one usable parsed page."""

    page_id: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class ChunkSpan:
    """Backend chunk output expressed in reconstructed-text coordinates."""

    content: str
    start_index: int
    end_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """RAGit-owned interface for document-level chunkers.

    Concrete implementations only chunk reconstructed text and return exact
    character spans. RAGit owns parsed-page reconstruction, page provenance,
    and normalized ``Chunk`` creation.
    """

    @classmethod
    def validate_config(cls, config: ChunkingConfig) -> None:
        """Validate configuration accepted by this chunker."""
        if not isinstance(config, ChunkingConfig):
            raise TypeError("config must be a ChunkingConfig instance.")

    def chunk_document(
        self,
        parsed_pages: Sequence[ParsedPage],
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Chunk one parsed document and preserve source page provenance."""
        self.validate_config(config)

        if not parsed_pages:
            return []

        doc_id = _validate_single_document(parsed_pages)
        document_text, page_ranges = _reconstruct_document(parsed_pages)

        if not document_text:
            return []

        spans = self._chunk_text(
            text=document_text,
            config=config,
        )

        chunks: list[Chunk] = []

        for span in spans:
            _validate_chunk_span(
                span=span,
                document_length=len(document_text),
            )

            page_ids = _page_ids_for_span(
                start_index=span.start_index,
                end_index=span.end_index,
                page_ranges=page_ranges,
            )

            # A backend may theoretically emit a chunk containing only the
            # neutral separator between pages. Such a chunk has no source
            # page and is not useful for page-level retrieval evaluation.
            if not page_ids:
                continue

            chunk_index = len(chunks)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{chunk_index:08d}",
                    doc_id=doc_id,
                    page_ids=page_ids,
                    content=span.content,
                    metadata=dict(span.metadata),
                )
            )

        return chunks

    @abstractmethod
    def _chunk_text(
        self,
        *,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        """Chunk reconstructed text and return exact character spans."""


def _validate_single_document(
    parsed_pages: Sequence[ParsedPage],
) -> str:
    """Require all parsed pages to belong to one document."""
    doc_ids = {page.doc_id for page in parsed_pages}

    if len(doc_ids) != 1:
        raise ValueError(
            "chunk_document expects parsed pages from exactly one document."
        )

    return next(iter(doc_ids))


def _reconstruct_document(
    parsed_pages: Sequence[ParsedPage],
) -> tuple[str, list[_PageRange]]:
    """Reconstruct document text while tracking original page ranges.

    Only successful pages with usable string content contribute text. Usable
    pages are joined with one newline. The separator is deliberately outside
    every page range and therefore belongs to no page.
    """
    parts: list[str] = []
    page_ranges: list[_PageRange] = []
    cursor = 0
    has_usable_page = False

    for page in sorted(parsed_pages, key=lambda item: item.page_number):
        if page.status != PageStatus.SUCCESS:
            continue

        if not isinstance(page.content, str) or not page.content:
            continue

        if has_usable_page:
            parts.append(_PAGE_SEPARATOR)
            cursor += len(_PAGE_SEPARATOR)

        start_index = cursor
        parts.append(page.content)
        cursor += len(page.content)

        page_ranges.append(
            _PageRange(
                page_id=page.page_id,
                start_index=start_index,
                end_index=cursor,
            )
        )
        has_usable_page = True

    return "".join(parts), page_ranges


def _page_ids_for_span(
    *,
    start_index: int,
    end_index: int,
    page_ranges: Sequence[_PageRange],
) -> list[int]:
    """Return every page whose half-open range overlaps a chunk span."""
    return [
        page_range.page_id
        for page_range in page_ranges
        if start_index < page_range.end_index
        and end_index > page_range.start_index
    ]


def _validate_chunk_span(
    *,
    span: ChunkSpan,
    document_length: int,
) -> None:
    """Validate a backend span against reconstructed document coordinates."""
    if not isinstance(span.content, str) or not span.content:
        raise ValueError("Chunker returned an empty chunk content value.")

    if not isinstance(span.start_index, int) or isinstance(span.start_index, bool):
        raise TypeError("Chunk start_index must be an integer.")

    if not isinstance(span.end_index, int) or isinstance(span.end_index, bool):
        raise TypeError("Chunk end_index must be an integer.")

    if span.start_index < 0:
        raise ValueError("Chunk start_index must be non-negative.")

    if span.end_index <= span.start_index:
        raise ValueError("Chunk end_index must be greater than start_index.")

    if span.end_index > document_length:
        raise ValueError(
            "Chunk end_index exceeds the reconstructed document length."
        )
