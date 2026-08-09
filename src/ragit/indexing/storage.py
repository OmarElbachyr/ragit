"""Deterministic persistence for RAGit indexes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragit.indexing.models import (
    DenseIndexResult,
    IndexedChunk,
    IndexingConfig,
    LateInteractionIndexResult,
)


CONFIG_FILE = "config.json"
DENSE_INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
LATE_INTERACTION_READY_FILE = "index.ready"
PYLATE_INDEX_NAME = "index"


class IndexingStorageError(ValueError):
    """Raised for invalid or inconsistent persisted indexing artifacts."""


def indexing_configuration_hash(
    config: IndexingConfig,
    chunking_config_hash: str,
) -> str:
    payload = _configuration_payload(
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise IndexingStorageError(
            "Indexing configuration must be JSON serializable for persistence."
        ) from error
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def indexing_output_path(
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
) -> Path:
    config_hash = indexing_configuration_hash(
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    return (
        Path(pdfs_path)
        / ".ragit"
        / "indexing"
        / config.index_type
        / config_hash
    )


# ---------------------------------------------------------------------------
# Dense persistence
# ---------------------------------------------------------------------------


def save_dense_index(
    *,
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
    index: Any,
    chunk_mapping: Iterable[IndexedChunk],
    replace: bool = False,
) -> DenseIndexResult:
    output_path = indexing_output_path(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    config_path = output_path / CONFIG_FILE
    index_path = output_path / DENSE_INDEX_FILE
    chunks_path = output_path / CHUNKS_FILE

    if not replace and any(
        path.exists() for path in (config_path, index_path, chunks_path)
    ):
        raise FileExistsError(
            f"Dense indexing result already exists: {output_path}"
        )

    mapping = _validate_dense_mapping(chunk_mapping)
    ntotal = getattr(index, "ntotal", None)
    if ntotal is None or int(ntotal) != len(mapping):
        raise IndexingStorageError(
            "FAISS index size does not match chunk-position mapping size."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        config_path,
        _configuration_payload(
            config=config,
            chunking_config_hash=chunking_config_hash,
        ),
    )
    _write_faiss_atomic(index_path, index)
    _write_jsonl_atomic(
        chunks_path,
        (item.model_dump(mode="json") for item in mapping),
    )

    return DenseIndexResult(
        index=index,
        chunk_mapping=mapping,
        config_hash=indexing_configuration_hash(
            config=config,
            chunking_config_hash=chunking_config_hash,
        ),
        output_path=output_path,
        config=config,
    )


def load_dense_index(
    *,
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
) -> DenseIndexResult:
    output_path = indexing_output_path(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    config_path = output_path / CONFIG_FILE
    index_path = output_path / DENSE_INDEX_FILE
    chunks_path = output_path / CHUNKS_FILE

    _require_files(
        output_path,
        (config_path, index_path, chunks_path),
        label="dense indexing",
    )
    _validate_saved_config(
        config_path=config_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )

    mapping: list[IndexedChunk] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                mapping.append(IndexedChunk.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as error:
                raise IndexingStorageError(
                    f"Invalid chunk mapping in {chunks_path} at line "
                    f"{line_number}: {error}"
                ) from error

    mapping = _validate_dense_mapping(mapping)
    faiss = _import_faiss()
    index = faiss.read_index(str(index_path))
    if int(index.ntotal) != len(mapping):
        raise IndexingStorageError(
            "Persisted FAISS index size does not match chunk mapping size."
        )

    return DenseIndexResult(
        index=index,
        chunk_mapping=mapping,
        config_hash=indexing_configuration_hash(
            config=config,
            chunking_config_hash=chunking_config_hash,
        ),
        output_path=output_path,
        config=config,
    )


def artifact_state(output_path: Path) -> str:
    """Backward-compatible dense artifact state helper."""
    return dense_artifact_state(output_path)


def dense_artifact_state(output_path: Path) -> str:
    return _artifact_state(
        (
            output_path / CONFIG_FILE,
            output_path / DENSE_INDEX_FILE,
            output_path / CHUNKS_FILE,
        )
    )


# ---------------------------------------------------------------------------
# Late-interaction persistence metadata
# ---------------------------------------------------------------------------


def save_late_interaction_index(
    *,
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
    index: Any,
    chunk_ids: Iterable[str],
    replace: bool = False,
) -> LateInteractionIndexResult:
    """Persist RAGit metadata after PyLate has persisted its own index."""
    output_path = indexing_output_path(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    config_path = output_path / CONFIG_FILE
    chunks_path = output_path / CHUNKS_FILE
    ready_path = output_path / LATE_INTERACTION_READY_FILE

    if not replace and any(
        path.exists() for path in (config_path, chunks_path, ready_path)
    ):
        raise FileExistsError(
            f"Late-interaction indexing result already exists: {output_path}"
        )

    normalized_ids = _validate_chunk_ids(chunk_ids)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        config_path,
        _configuration_payload(
            config=config,
            chunking_config_hash=chunking_config_hash,
        ),
    )
    _write_jsonl_atomic(
        chunks_path,
        ({"chunk_id": chunk_id} for chunk_id in normalized_ids),
    )
    _write_text_atomic(ready_path, "ready\n")

    return LateInteractionIndexResult(
        index=index,
        chunk_ids=normalized_ids,
        config_hash=indexing_configuration_hash(
            config=config,
            chunking_config_hash=chunking_config_hash,
        ),
        output_path=output_path,
        config=config,
    )


def load_late_interaction_metadata(
    *,
    pdfs_path: str | Path,
    config: IndexingConfig,
    chunking_config_hash: str,
) -> tuple[Path, list[str], str]:
    """Validate/load RAGit metadata for a PyLate index."""
    output_path = indexing_output_path(
        pdfs_path=pdfs_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    config_path = output_path / CONFIG_FILE
    chunks_path = output_path / CHUNKS_FILE
    ready_path = output_path / LATE_INTERACTION_READY_FILE

    _require_files(
        output_path,
        (config_path, chunks_path, ready_path),
        label="late-interaction indexing",
    )
    _validate_saved_config(
        config_path=config_path,
        config=config,
        chunking_config_hash=chunking_config_hash,
    )

    chunk_ids: list[str] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                chunk_id = record["chunk_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise IndexingStorageError(
                    f"Invalid late-interaction chunk mapping in {chunks_path} "
                    f"at line {line_number}."
                ) from error
            chunk_ids.append(chunk_id)

    return (
        output_path,
        _validate_chunk_ids(chunk_ids),
        indexing_configuration_hash(
            config=config,
            chunking_config_hash=chunking_config_hash,
        ),
    )


def late_interaction_artifact_state(output_path: Path) -> str:
    return _artifact_state(
        (
            output_path / CONFIG_FILE,
            output_path / CHUNKS_FILE,
            output_path / LATE_INTERACTION_READY_FILE,
        )
    )


def clear_late_interaction_ready_marker(output_path: Path) -> None:
    """Prevent a failed rebuild from looking like a complete artifact."""
    ready_path = output_path / LATE_INTERACTION_READY_FILE
    if ready_path.exists():
        ready_path.unlink()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _configuration_payload(
    *,
    config: IndexingConfig,
    chunking_config_hash: str,
) -> dict[str, Any]:
    if not isinstance(chunking_config_hash, str) or not chunking_config_hash:
        raise IndexingStorageError(
            "chunking_config_hash must be a non-empty string."
        )
    return {
        "indexing_config": config.model_dump(mode="json"),
        "chunking_config_hash": chunking_config_hash,
    }


def _validate_saved_config(
    *,
    config_path: Path,
    config: IndexingConfig,
    chunking_config_hash: str,
) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        saved_payload = json.load(file)
    expected_payload = _configuration_payload(
        config=config,
        chunking_config_hash=chunking_config_hash,
    )
    if saved_payload != expected_payload:
        raise IndexingStorageError(
            "Saved indexing configuration does not match the requested "
            "indexing/chunking configuration."
        )


def _validate_dense_mapping(
    chunk_mapping: Iterable[IndexedChunk],
) -> list[IndexedChunk]:
    mapping = list(chunk_mapping)
    seen_chunk_ids: set[str] = set()
    for expected_position, item in enumerate(mapping):
        if not isinstance(item, IndexedChunk):
            raise TypeError("chunk_mapping must contain IndexedChunk instances.")
        if item.index_position != expected_position:
            raise IndexingStorageError(
                "Chunk mapping positions must be contiguous and preserve "
                "FAISS insertion order."
            )
        if item.chunk_id in seen_chunk_ids:
            raise IndexingStorageError(
                f"Duplicate chunk_id in index mapping: {item.chunk_id!r}."
            )
        seen_chunk_ids.add(item.chunk_id)
    return mapping


def _validate_chunk_ids(chunk_ids: Iterable[str]) -> list[str]:
    normalized = list(chunk_ids)
    seen: set[str] = set()
    for chunk_id in normalized:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise IndexingStorageError(
                "Late-interaction chunk IDs must be non-empty strings."
            )
        if chunk_id in seen:
            raise IndexingStorageError(
                f"Duplicate chunk_id in late-interaction index: {chunk_id!r}."
            )
        seen.add(chunk_id)
    return normalized


def _artifact_state(paths: Iterable[Path]) -> str:
    existence = [path.exists() for path in paths]
    if all(existence):
        return "complete"
    if any(existence):
        return "partial"
    return "missing"


def _require_files(
    output_path: Path,
    paths: Iterable[Path],
    *,
    label: str,
) -> None:
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {label} artifact at {output_path}. Missing: {missing}."
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    temporary_path.replace(path)


def _write_jsonl_atomic(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def _write_faiss_atomic(path: Path, index: Any) -> None:
    faiss = _import_faiss()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary_path))
    temporary_path.replace(path)


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise ImportError(
            "faiss-cpu is required for dense RAGit indexing."
        ) from error
    return faiss
