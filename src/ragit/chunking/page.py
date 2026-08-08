"""Native page-level chunker for RAGit."""

from __future__ import annotations

from collections.abc import Sequence

from ragit.parsing.models import PageStatus, ParsedPage

from ragit.chunking.base import BaseChunker, ChunkSpan, _validate_single_document
from ragit.chunking.models import Chunk, ChunkingConfig
from ragit.chunking.registry import register_chunker


class PageChunker(BaseChunker):
    """Create exactly one chunk for each usable parsed page."""

    chunker_name = "page"

    @classmethod
    def validate_config(cls, config: ChunkingConfig) -> None:
        """Accept the native page strategy with no options."""
        super().validate_config(config)

        if config.chunker_name != cls.chunker_name:
            raise ValueError(
                f"PageChunker requires chunker_name={cls.chunker_name!r}."
            )

        if config.options:
            raise ValueError(
                "Unsupported options for 'page': "
                f"{sorted(config.options)}. The page chunker accepts no options."
            )

    def chunk_document(
        self,
        parsed_pages: Sequence[ParsedPage],
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Return one normalized chunk per successful non-empty page."""
        self.validate_config(config)

        if not parsed_pages:
            return []

        doc_id = _validate_single_document(parsed_pages)
        chunks: list[Chunk] = []

        for page in sorted(parsed_pages, key=lambda item: item.page_number):
            if page.status != PageStatus.SUCCESS:
                continue

            if not isinstance(page.content, str) or not page.content:
                continue

            chunk_index = len(chunks)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{chunk_index:08d}",
                    doc_id=doc_id,
                    page_ids=[page.page_id],
                    content=page.content,
                    metadata={
                        "backend": "ragit",
                        "chunker_name": "page",
                    },
                )
            )

        return chunks

    def _chunk_text(
        self,
        *,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        """Unused because page chunking operates directly on ParsedPage objects."""
        raise NotImplementedError(
            "PageChunker chunks ParsedPage objects directly."
        )


register_chunker(PageChunker)
