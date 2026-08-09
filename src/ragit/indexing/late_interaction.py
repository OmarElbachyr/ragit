"""PyLate ColBERT-style multi-vector indexing for RAGit."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ragit.chunking.models import Chunk, ChunkingResult
from ragit.data.models import Collection
from ragit.indexing.models import IndexingConfig, LateInteractionIndexResult
from ragit.indexing.storage import (
    IndexingStorageError,
    PYLATE_INDEX_NAME,
    clear_late_interaction_ready_marker,
    indexing_output_path,
    load_late_interaction_metadata,
    save_late_interaction_index,
)
from ragit.indexing.validation import (
    IndexingConfigurationError,
    validate_chunks,
    validate_late_interaction_config,
    validate_late_interaction_encoder,
    validate_multi_vector_embeddings,
)


def _build_late_interaction_index_from_result(
    *,
    collection: Collection,
    chunking_result: ChunkingResult,
    config: IndexingConfig,
    encoder: Any | None = None,
) -> LateInteractionIndexResult:
    """Build a PyLate PLAID index from normalized RAGit chunks."""
    if not isinstance(chunking_result, ChunkingResult):
        raise TypeError("chunking_result must be a ChunkingResult instance.")

    result = _build_late_interaction_index(
        chunks=chunking_result.chunks,
        pdfs_path=collection.pdfs_path,
        config=config,
        chunking_config_hash=chunking_result.config_hash,
        encoder=encoder,
    )
    return result.model_copy(
        update={
            "config": config,
            "chunks": list(chunking_result.chunks),
        }
    )


def _load_late_interaction_index_from_result(
    *,
    collection: Collection,
    chunking_result: ChunkingResult,
    config: IndexingConfig,
) -> LateInteractionIndexResult:
    """Load a persisted PyLate index and attach originating chunks."""
    if not isinstance(chunking_result, ChunkingResult):
        raise TypeError("chunking_result must be a ChunkingResult instance.")

    validate_late_interaction_config(config)
    output_path, chunk_ids, config_hash = load_late_interaction_metadata(
        pdfs_path=collection.pdfs_path,
        config=config,
        chunking_config_hash=chunking_result.config_hash,
    )

    expected_ids = [chunk.chunk_id for chunk in chunking_result.chunks]
    if chunk_ids != expected_ids:
        raise IndexingStorageError(
            "Persisted late-interaction chunk IDs do not match the current "
            "chunking result."
        )

    index = _create_pylate_index(
        output_path=output_path,
        config=config,
        override=False,
    )
    return LateInteractionIndexResult(
        index=index,
        chunk_ids=chunk_ids,
        config_hash=config_hash,
        output_path=output_path,
        config=config,
        chunks=list(chunking_result.chunks),
    )


def _build_late_interaction_index(
    *,
    chunks: Iterable[Chunk],
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
    encoder: Any | None,
) -> LateInteractionIndexResult:
    validate_late_interaction_config(config)

    output_path = indexing_output_path(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    normalized_chunks = validate_chunks(chunks)
    if not normalized_chunks:
        raise IndexingConfigurationError(
            "Late-interaction indexing requires at least one chunk."
        )

    if encoder is None:
        encoder = _load_encoder(config)
    validate_late_interaction_encoder(encoder, config.model_name)

    document_embeddings = _encode_chunks(
        encoder=encoder,
        chunks=normalized_chunks,
        config=config,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    clear_late_interaction_ready_marker(output_path)
    index = _create_pylate_index(
        output_path=output_path,
        config=config,
        override=True,
    )

    chunk_ids = [chunk.chunk_id for chunk in normalized_chunks]
    try:
        index.add_documents(
            documents_ids=chunk_ids,
            documents_embeddings=document_embeddings,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise IndexingConfigurationError(
            "PyLate failed while adding chunk multi-vector representations "
            "to the PLAID index."
        ) from error

    return save_late_interaction_index(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
        index=index,
        chunk_ids=chunk_ids,
        replace=True,
    )


def _encode_chunks(
    *,
    encoder: Any,
    chunks: list[Chunk],
    config: IndexingConfig,
) -> list[Any]:
    """Encode one variable-length token matrix per chunk with PyLate."""
    options = config.options
    encode_kwargs: dict[str, Any] = {
        "batch_size": options.get("batch_size", 32),
        "is_query": False,
        "show_progress_bar": options.get("show_progress_bar", False),
    }
    pool_factor = options.get("pool_factor")
    if pool_factor is not None:
        encode_kwargs["pool_factor"] = pool_factor

    try:
        embeddings = encoder.encode(
            [chunk.content for chunk in chunks],
            **encode_kwargs,
        )
    except TypeError as error:
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} does not provide a compatible "
            "PyLate ColBERT encode() interface."
        ) from error

    return validate_multi_vector_embeddings(
        embeddings,
        expected_count=len(chunks),
        model_name=config.model_name,
    )


def _load_encoder(config: IndexingConfig) -> Any:
    try:
        from pylate import models
    except ImportError as error:
        raise ImportError(
            "pylate is required for late-interaction RAGit indexing."
        ) from error

    kwargs: dict[str, Any] = {
        "model_name_or_path": config.model_name,
    }
    device = config.options.get("device")
    if device is not None:
        kwargs["device"] = device
    if config.options.get("trust_remote_code", False):
        kwargs["trust_remote_code"] = True

    return models.ColBERT(**kwargs)


def _create_pylate_index(
    *,
    output_path: Path,
    config: IndexingConfig,
    override: bool,
) -> Any:
    try:
        from pylate import indexes
    except ImportError as error:
        raise ImportError(
            "pylate is required for late-interaction RAGit indexing."
        ) from error

    kwargs: dict[str, Any] = {
        "index_folder": str(output_path),
        "index_name": PYLATE_INDEX_NAME,
        "override": override,
    }
    device = config.options.get("device")
    if device is not None:
        kwargs["device"] = device

    return indexes.PLAID(**kwargs)
