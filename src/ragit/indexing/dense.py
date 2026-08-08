"""Dense single-vector indexing baseline for RAGit."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ragit.chunking.models import Chunk, ChunkingResult
from ragit.data.models import Collection
from ragit.indexing.models import DenseIndexResult, IndexedChunk, IndexingConfig
from ragit.indexing.storage import (
    IndexingStorageError,
    artifact_state,
    indexing_output_path,
    load_dense_index as _load_dense_index,
    save_dense_index,
)
from ragit.indexing.validation import (
    IndexingConfigurationError,
    validate_chunks,
    validate_dense_config,
    validate_dense_encoder,
)


def _build_dense_index_from_result(
    *,
    collection: Collection,
    chunking_result: ChunkingResult,
    config: IndexingConfig,
    overwrite: bool = False,
    encoder: Any | None = None,
) -> DenseIndexResult:
    """Build or reuse a dense index from one normalized chunking result.

    Collection location and chunking provenance are derived from their result
    objects so callers do not need to pass storage paths or configuration
    hashes manually.

    ``encoder`` is primarily useful for tests or callers that already own a
    compatible SentenceTransformer-style model. When omitted, RAGit loads
    ``config.model_name`` lazily with Sentence Transformers.
    """
    if not isinstance(chunking_result, ChunkingResult):
        raise TypeError("chunking_result must be a ChunkingResult instance.")

    result = _build_dense_index(
        chunks=chunking_result.chunks,
        pdfs_path=collection.pdfs_path,
        config=config,
        chunking_config_hash=chunking_result.config_hash,
        overwrite=overwrite,
        encoder=encoder,
    )
    return result.model_copy(
        update={
            "config": config,
            "chunks": list(chunking_result.chunks),
        }
    )


def _load_dense_index_from_result(
    *,
    collection: Collection,
    chunking_result: ChunkingResult,
    config: IndexingConfig,
) -> DenseIndexResult:
    """Load a dense index derived from one normalized chunking result."""
    if not isinstance(chunking_result, ChunkingResult):
        raise TypeError("chunking_result must be a ChunkingResult instance.")

    validate_dense_config(config)

    result = _load_dense_index(
        pdfs_path=collection.pdfs_path,
        config=config,
        chunking_config_hash=chunking_result.config_hash,
    )
    return result.model_copy(
        update={
            "config": config,
            "chunks": list(chunking_result.chunks),
        }
    )


def _build_dense_index(
    *,
    chunks: Iterable[Chunk],
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
    overwrite: bool = False,
    encoder: Any | None = None,
) -> DenseIndexResult:
    """Internal dense builder using explicit storage/provenance arguments."""
    validate_dense_config(config)

    output_path = indexing_output_path(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    state = artifact_state(output_path)

    if state == "complete" and not overwrite:
        return _load_dense_index(
            pdfs_path=pdfs_path,
            config=config,
            chunking_config_hash=chunking_config_hash,
        )

    if state == "partial" and not overwrite:
        raise IndexingStorageError(
            f"Incomplete dense indexing artifact exists at {output_path}. "
            "Use overwrite=True to rebuild it."
        )

    normalized_chunks = validate_chunks(chunks)
    if not normalized_chunks:
        raise IndexingConfigurationError(
            "Dense indexing requires at least one chunk."
        )

    if encoder is None:
        encoder = _load_encoder(config)

    expected_dimension = validate_dense_encoder(
        encoder=encoder,
        model_name=config.model_name,
    )

    embeddings = _encode_chunks(
        encoder=encoder,
        chunks=normalized_chunks,
        config=config,
        expected_dimension=expected_dimension,
    )

    faiss = _import_faiss()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    mapping = [
        IndexedChunk(
            index_position=position,
            chunk_id=chunk.chunk_id,
        )
        for position, chunk in enumerate(normalized_chunks)
    ]

    return save_dense_index(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
        index=index,
        chunk_mapping=mapping,
        replace=overwrite or state != "missing",
    )


def _encode_chunks(
    *,
    encoder: Any,
    chunks: list[Chunk],
    config: IndexingConfig,
    expected_dimension: int,
) -> np.ndarray:
    """Encode chunks, enforce dense shape/float32, and L2-normalize."""
    texts = [chunk.content for chunk in chunks]
    options = config.options

    encode_kwargs = {
        "batch_size": options.get("batch_size", 32),
        "show_progress_bar": options.get("show_progress_bar", False),
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }

    try:
        raw_embeddings = encoder.encode(texts, **encode_kwargs)
    except TypeError as error:
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} does not provide a compatible "
            "SentenceTransformer-style encode() interface."
        ) from error

    try:
        embeddings = np.asarray(raw_embeddings)
    except (TypeError, ValueError) as error:
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} did not return a rectangular "
            "single-vector embedding matrix."
        ) from error

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} produces multi-vector or otherwise "
            "non-dense representations and is not compatible with "
            "index_type='dense'. Expected shape "
            f"({len(chunks)}, dimension), got {embeddings.shape}."
        )

    if embeddings.shape[1] != expected_dimension:
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} advertised embedding dimension "
            f"{expected_dimension}, but encode() returned "
            f"{embeddings.shape[1]}."
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise IndexingConfigurationError(
            f"Model {config.model_name!r} produced at least one zero vector, "
            "which cannot be L2-normalized."
        )

    embeddings /= norms
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def _load_encoder(config: IndexingConfig) -> Any:
    """Load the configured dense model through Sentence Transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required to load dense embedding "
            "models by name. Install it or pass an already loaded compatible "
            "encoder to build_index()."
        ) from error

    kwargs: dict[str, Any] = {}
    device = config.options.get("device")
    if device is not None:
        kwargs["device"] = device

    return SentenceTransformer(config.model_name, **kwargs)


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise ImportError(
            "faiss-cpu is required for dense RAGit indexing."
        ) from error
    return faiss
