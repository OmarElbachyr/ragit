"""Public indexing interfaces."""

from ragit.indexing.api import build_index, load_index
from ragit.indexing.models import (
    DenseIndexResult,
    IndexedChunk,
    IndexingConfig,
    IndexResult,
    LateInteractionIndexResult,
)
from ragit.indexing.storage import IndexingStorageError
from ragit.indexing.validation import IndexingConfigurationError

__all__ = [
    "DenseIndexResult",
    "IndexedChunk",
    "IndexingConfig",
    "IndexResult",
    "IndexingConfigurationError",
    "IndexingStorageError",
    "LateInteractionIndexResult",
    "build_index",
    "load_index",
]
