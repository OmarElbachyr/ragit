"""Public indexing dispatcher."""

from __future__ import annotations

from typing import Any

from ragit.chunking.models import ChunkingResult
from ragit.data.models import Collection
from ragit.indexing.dense import (
    _build_dense_index_from_result,
    _load_dense_index_from_result,
)
from ragit.indexing.late_interaction import (
    _build_late_interaction_index_from_result,
    _load_late_interaction_index_from_result,
)
from ragit.indexing.models import IndexResult, IndexingConfig


def build_index(
    *,
    collection: Collection,
    chunking_result: ChunkingResult,
    config: IndexingConfig,
    encoder: Any | None = None,
) -> IndexResult:
    """Build the index selected by ``config.index_type``."""
    if config.index_type == "dense":
        return _build_dense_index_from_result(
            collection=collection,
            chunking_result=chunking_result,
            config=config,
            encoder=encoder,
        )

    if config.index_type == "late_interaction":
        return _build_late_interaction_index_from_result(
            collection=collection,
            chunking_result=chunking_result,
            config=config,
            encoder=encoder,
        )

    raise ValueError(f"Unsupported index_type: {config.index_type!r}.")


def load_index(
    *,
    collection: Collection,
    chunking_result: ChunkingResult,
    config: IndexingConfig,
) -> IndexResult:
    """Load the index selected by ``config.index_type``."""
    if config.index_type == "dense":
        return _load_dense_index_from_result(
            collection=collection,
            chunking_result=chunking_result,
            config=config,
        )

    if config.index_type == "late_interaction":
        return _load_late_interaction_index_from_result(
            collection=collection,
            chunking_result=chunking_result,
            config=config,
        )

    raise ValueError(f"Unsupported index_type: {config.index_type!r}.")
