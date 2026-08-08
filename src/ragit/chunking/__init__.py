"""Public chunking interfaces."""

from ragit.chunking.base import BaseChunker, ChunkSpan
from ragit.chunking.models import Chunk, ChunkingConfig, ChunkingResult
from ragit.chunking.registry import (
    ChunkerRegistryError,
    get_chunker,
    register_chunker,
)

# Import built-in chunkers so their registry entries are available to the
# public API. Concrete implementations remain internal details.
from ragit.chunking import chonkie as _chonkie  # noqa: F401
from ragit.chunking import page as _page  # noqa: F401

from ragit.chunking.api import chunk_document
from ragit.chunking.storage import (
    chunking_configuration_hash,
    chunking_output_path,
    load_chunking_result,
    save_chunking_result,
)

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkSpan",
    "ChunkerRegistryError",
    "ChunkingConfig",
    "ChunkingResult",
    "chunk_document",
    "chunking_configuration_hash",
    "chunking_output_path",
    "get_chunker",
    "load_chunking_result",
    "register_chunker",
    "save_chunking_result",
]
