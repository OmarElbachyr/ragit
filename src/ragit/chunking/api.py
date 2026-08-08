"""High-level public API for RAGit chunking."""

from __future__ import annotations

from collections.abc import Sequence

from ragit.parsing.models import ParsedPage

from ragit.chunking.models import Chunk, ChunkingConfig
from ragit.chunking.registry import get_chunker


def chunk_document(
    parsed_pages: Sequence[ParsedPage],
    config: ChunkingConfig,
) -> list[Chunk]:
    """Chunk one parsed document using the configured implementation.

    The selected component is resolved through the RAGit chunker registry.
    Each concrete chunker owns validation and execution. Text-based chunkers
    can use the shared BaseChunker reconstruction/provenance path, while native
    strategies such as ``page`` may operate directly on ParsedPage objects.
    """
    if not isinstance(config, ChunkingConfig):
        raise TypeError("config must be a ChunkingConfig instance.")

    chunker_class = get_chunker(config.chunker_name)
    chunker = chunker_class()

    chunker.validate_config(config)

    return chunker.chunk_document(
        parsed_pages=parsed_pages,
        config=config,
    )
