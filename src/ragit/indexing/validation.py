"""Validation helpers for RAGit indexing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ragit.chunking.models import Chunk
from ragit.indexing.models import IndexingConfig


_DENSE_OPTIONS = frozenset({"batch_size", "device", "show_progress_bar"})
_LATE_INTERACTION_OPTIONS = frozenset(
    {
        "batch_size",
        "device",
        "show_progress_bar",
        "trust_remote_code",
        "pool_factor",
    }
)
_MULTI_VECTOR_CLASS_MARKERS = ("multivector", "lateinteraction", "colbert")


class IndexingConfigurationError(ValueError):
    """Raised when an indexing configuration is incompatible or invalid."""


def validate_dense_config(config: IndexingConfig) -> None:
    """Validate dense single-vector indexing options."""
    _validate_config_instance(config)
    if config.index_type != "dense":
        raise IndexingConfigurationError(
            f"Dense indexing requires index_type='dense', got "
            f"{config.index_type!r}."
        )
    _validate_common_options(config, _DENSE_OPTIONS, "dense")


def validate_late_interaction_config(config: IndexingConfig) -> None:
    """Validate PyLate late-interaction indexing options."""
    _validate_config_instance(config)
    if config.index_type != "late_interaction":
        raise IndexingConfigurationError(
            "Late-interaction indexing requires "
            "index_type='late_interaction'."
        )
    _validate_common_options(
        config,
        _LATE_INTERACTION_OPTIONS,
        "late-interaction",
    )

    trust_remote_code = config.options.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise IndexingConfigurationError(
            "The 'trust_remote_code' option must be a boolean."
        )

    pool_factor = config.options.get("pool_factor")
    if pool_factor is not None:
        if isinstance(pool_factor, bool) or not isinstance(pool_factor, int):
            raise IndexingConfigurationError(
                "The 'pool_factor' option must be an integer."
            )
        if pool_factor <= 0:
            raise IndexingConfigurationError(
                "The 'pool_factor' option must be greater than zero."
            )


def validate_dense_encoder(encoder: Any, model_name: str) -> int:
    """Validate that an encoder advertises one fixed-size vector per input."""
    if _has_multi_vector_signal(encoder, include_self=False):
        raise IndexingConfigurationError(
            f"Model {model_name!r} produces multi-vector representations "
            "and is not compatible with index_type='dense'."
        )

    dimension = _get_embedding_dimension(encoder)
    if dimension is None:
        raise IndexingConfigurationError(
            f"Model {model_name!r} does not expose a fixed single-vector "
            "embedding dimension and cannot be validated for "
            "index_type='dense'."
        )

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise IndexingConfigurationError(
            f"Model {model_name!r} returned an invalid embedding dimension: "
            f"{dimension!r}."
        )
    if dimension <= 0:
        raise IndexingConfigurationError(
            f"Model {model_name!r} returned a non-positive embedding "
            f"dimension: {dimension}."
        )
    if not callable(getattr(encoder, "encode", None)):
        raise IndexingConfigurationError(
            f"Model {model_name!r} does not provide an encode() method."
        )
    return dimension


def validate_late_interaction_encoder(encoder: Any, model_name: str) -> None:
    """Validate a ColBERT-style multi-vector encoder before corpus encoding."""
    if not callable(getattr(encoder, "encode", None)):
        raise IndexingConfigurationError(
            f"Model {model_name!r} does not provide an encode() method."
        )
    if not _has_multi_vector_signal(encoder, include_self=True):
        raise IndexingConfigurationError(
            f"Model {model_name!r} does not expose a ColBERT/late-interaction "
            "multi-vector capability and is not compatible with "
            "index_type='late_interaction'."
        )


def validate_multi_vector_embeddings(
    embeddings: Any,
    *,
    expected_count: int,
    model_name: str,
) -> list[Any]:
    """Validate one variable-length token matrix per encoded input."""
    try:
        normalized = list(embeddings)
    except TypeError as error:
        raise IndexingConfigurationError(
            f"Model {model_name!r} did not return multi-vector "
            "representations."
        ) from error

    if len(normalized) != expected_count:
        raise IndexingConfigurationError(
            f"Model {model_name!r} returned {len(normalized)} representations "
            f"for {expected_count} inputs."
        )

    embedding_dimension: int | None = None
    for position, representation in enumerate(normalized):
        shape = getattr(representation, "shape", None)
        if shape is None or len(shape) != 2:
            raise IndexingConfigurationError(
                f"Model {model_name!r} is not compatible with "
                "index_type='late_interaction'. Expected one 2D token "
                f"embedding matrix per input; item {position} has shape "
                f"{shape!r}."
            )
        token_count, dimension = int(shape[0]), int(shape[1])
        if token_count <= 0 or dimension <= 0:
            raise IndexingConfigurationError(
                f"Model {model_name!r} returned an empty/invalid multi-vector "
                f"representation at item {position}: {shape!r}."
            )
        if embedding_dimension is None:
            embedding_dimension = dimension
        elif dimension != embedding_dimension:
            raise IndexingConfigurationError(
                f"Model {model_name!r} returned inconsistent token embedding "
                "dimensions across inputs."
            )

    return normalized


def validate_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Validate chunk IDs/content while preserving caller insertion order."""
    normalized = list(chunks)
    seen_ids: set[str] = set()

    for chunk in normalized:
        if not isinstance(chunk, Chunk):
            raise TypeError("chunks must contain Chunk instances.")
        if chunk.chunk_id in seen_ids:
            raise IndexingConfigurationError(
                f"Duplicate chunk_id in indexing input: {chunk.chunk_id!r}."
            )
        if not isinstance(chunk.content, str) or not chunk.content:
            raise IndexingConfigurationError(
                f"Chunk {chunk.chunk_id!r} has empty content."
            )
        seen_ids.add(chunk.chunk_id)

    return normalized


def _validate_config_instance(config: IndexingConfig) -> None:
    if not isinstance(config, IndexingConfig):
        raise TypeError("config must be an IndexingConfig instance.")


def _validate_common_options(
    config: IndexingConfig,
    allowed: frozenset[str],
    label: str,
) -> None:
    unsupported = set(config.options) - allowed
    if unsupported:
        raise IndexingConfigurationError(
            f"Unsupported {label} indexing options: {sorted(unsupported)}. "
            f"Supported options: {sorted(allowed)}."
        )

    batch_size = config.options.get("batch_size", 32)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise IndexingConfigurationError(
            "The 'batch_size' option must be an integer."
        )
    if batch_size <= 0:
        raise IndexingConfigurationError(
            "The 'batch_size' option must be greater than zero."
        )

    device = config.options.get("device")
    if device is not None and (
        not isinstance(device, str) or not device.strip()
    ):
        raise IndexingConfigurationError(
            "The 'device' option must be a non-empty string."
        )

    show_progress = config.options.get("show_progress_bar", False)
    if not isinstance(show_progress, bool):
        raise IndexingConfigurationError(
            "The 'show_progress_bar' option must be a boolean."
        )


def _get_embedding_dimension(encoder: Any) -> int | None:
    for method_name in (
        "get_embedding_dimension",
        "get_sentence_embedding_dimension",
    ):
        method = getattr(encoder, method_name, None)
        if callable(method):
            try:
                value = method()
            except (TypeError, ValueError):
                return None
            if value is not None:
                return value
    return None


def _has_multi_vector_signal(encoder: Any, *, include_self: bool) -> bool:
    for attribute in (
        "is_multi_vector",
        "multi_vector",
        "output_is_multi_vector",
    ):
        if getattr(encoder, attribute, False) is True:
            return True

    similarity_name = getattr(encoder, "similarity_fn_name", None)
    if isinstance(similarity_name, str) and similarity_name.lower() == "maxsim":
        return True

    config = getattr(encoder, "config", None)
    if config is not None:
        similarity_name = getattr(config, "similarity_fn_name", None)
        if isinstance(similarity_name, str) and similarity_name.lower() == "maxsim":
            return True
        for attribute in (
            "is_multi_vector",
            "multi_vector",
            "output_is_multi_vector",
        ):
            if getattr(config, attribute, False) is True:
                return True

    objects = list(_iter_modules(encoder))
    if include_self:
        objects.insert(0, encoder)
    for item in objects:
        class_name = item.__class__.__name__.replace("_", "").lower()
        if any(marker in class_name for marker in _MULTI_VECTOR_CLASS_MARKERS):
            return True
    return False


def _iter_modules(encoder: Any) -> Iterable[Any]:
    modules_method = getattr(encoder, "modules", None)
    if callable(modules_method):
        try:
            yield from modules_method()
            return
        except TypeError:
            pass

    try:
        yield from encoder
    except TypeError:
        return
