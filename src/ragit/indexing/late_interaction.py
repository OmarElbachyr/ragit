"""Sentence Transformers multi-vector encoding with FastPlaid indexing."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ragit.chunking.models import Chunk, ChunkingResult
from ragit.data.models import Collection
from ragit.indexing.models import IndexingConfig, LateInteractionIndexResult
from ragit.indexing.storage import (
    IndexingStorageError,
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
    """Build a FastPlaid index from normalized RAGit chunks."""
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
    """Load a persisted FastPlaid index and attach originating chunks."""
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

    index = _create_fast_plaid_index(
        output_path=output_path,
        config=config,
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
    index = _create_fast_plaid_index(
        output_path=output_path,
        config=config,
    )

    chunk_ids = [chunk.chunk_id for chunk in normalized_chunks]
    try:
        index.create(documents_embeddings=document_embeddings)
    except (TypeError, ValueError, RuntimeError) as error:
        raise IndexingConfigurationError(
            "FastPlaid failed while creating the late-interaction index."
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
    """Encode one variable-length token matrix per chunk for PLAID."""
    options = config.options
    encode_kwargs: dict[str, Any] = {
        "batch_size": options.get("batch_size", 32),
        "show_progress_bar": options.get("show_progress_bar", False),
    }

    try:
        embeddings = encoder.encode_document(
            [chunk.content for chunk in chunks],
            **encode_kwargs,
        )
    except TypeError as error:
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} does not provide a compatible "
            "Sentence Transformers MultiVectorEncoder document interface."
        ) from error

    return validate_multi_vector_embeddings(
        embeddings,
        expected_count=len(chunks),
        model_name=config.model_name,
    )


def _load_encoder(config: IndexingConfig) -> Any:
    try:
        from sentence_transformers import MultiVectorEncoder
    except ImportError as error:
        raise ImportError(
            "sentence-transformers>=6 is required to load late-interaction "
            "models by name. Install it or pass an already loaded compatible "
            "encoder to build_index()."
        ) from error

    kwargs: dict[str, Any] = {}
    device = config.options.get("device")
    if device is not None:
        kwargs["device"] = device
    if config.options.get("trust_remote_code", False):
        kwargs["trust_remote_code"] = True

    return MultiVectorEncoder(config.model_name, **kwargs)


def _create_fast_plaid_index(
    *,
    output_path: Path,
    config: IndexingConfig,
) -> Any:
    try:
        from fast_plaid import search
    except ImportError as error:
        raise ImportError(
            "fast-plaid is required for late-interaction RAGit indexing."
        ) from error

    kwargs: dict[str, Any] = {
        "index": str(output_path),
    }
    device = config.options.get("device")
    if device is not None:
        kwargs["device"] = device

    return search.FastPlaid(**kwargs)
