"""Deterministic persistence for normalized RAGit chunks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragit.chunking.models import Chunk, ChunkingConfig, ChunkingResult


CONFIG_FILE = "config.json"
CHUNKS_FILE = "chunks.jsonl"


def chunking_configuration_hash(
    config: ChunkingConfig,
    parsing_config_hash: str,
) -> str:
    """Hash chunking config together with its source parsing configuration."""
    payload = _configuration_payload(
        config=config,
        parsing_config_hash=parsing_config_hash,
    )

    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Chunking configuration must be JSON serializable for caching."
        ) from error

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def chunking_output_path(
    pdfs_path: str | Path,
    config: ChunkingConfig,
    parsing_config_hash: str,
) -> Path:
    """Return the deterministic directory for one chunking artifact."""
    pdfs_path = Path(pdfs_path)
    config_hash = chunking_configuration_hash(
        config=config,
        parsing_config_hash=parsing_config_hash,
    )

    return (
        pdfs_path
        / ".ragit"
        / "chunking"
        / config.chunker_name
        / config_hash
    )


def save_chunking_result(
    *,
    pdfs_path: str | Path,
    config: ChunkingConfig,
    parsing_config_hash: str,
    chunks: Iterable[Chunk],
    replace: bool = False,
) -> ChunkingResult:
    """Persist one normalized chunking result atomically per file."""
    output_path = chunking_output_path(
        pdfs_path=pdfs_path,
        config=config,
        parsing_config_hash=parsing_config_hash,
    )
    config_path = output_path / CONFIG_FILE
    chunks_path = output_path / CHUNKS_FILE

    if not replace and (config_path.exists() or chunks_path.exists()):
        raise FileExistsError(
            f"Chunking result already exists: {output_path}"
        )

    normalized_chunks = _validate_and_sort_chunks(chunks)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        config_path,
        _configuration_payload(
            config=config,
            parsing_config_hash=parsing_config_hash,
        ),
    )
    _write_jsonl_atomic(
        chunks_path,
        (chunk.model_dump(mode="json") for chunk in normalized_chunks),
    )

    return ChunkingResult(
        chunks=normalized_chunks,
        config_hash=chunking_configuration_hash(
            config=config,
            parsing_config_hash=parsing_config_hash,
        ),
        parsing_config_hash=parsing_config_hash,
        output_path=output_path,
    )


def load_chunking_result(
    *,
    pdfs_path: str | Path,
    config: ChunkingConfig,
    parsing_config_hash: str,
) -> ChunkingResult:
    """Load a persisted chunking result and validate its cache identity."""
    output_path = chunking_output_path(
        pdfs_path=pdfs_path,
        config=config,
        parsing_config_hash=parsing_config_hash,
    )
    config_path = output_path / CONFIG_FILE
    chunks_path = output_path / CHUNKS_FILE

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing chunking config file: {config_path}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing chunking chunks file: {chunks_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        saved_payload = json.load(file)

    expected_payload = _configuration_payload(
        config=config,
        parsing_config_hash=parsing_config_hash,
    )

    if saved_payload != expected_payload:
        raise ValueError(
            "Saved chunking configuration does not match the requested "
            "chunking/parsing configuration."
        )

    chunks: list[Chunk] = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                chunk = Chunk.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(
                    f"Invalid chunk record in {chunks_path} at line "
                    f"{line_number}: {error}"
                ) from error

            chunks.append(chunk)

    normalized_chunks = _validate_and_sort_chunks(chunks)

    return ChunkingResult(
        chunks=normalized_chunks,
        config_hash=chunking_configuration_hash(
            config=config,
            parsing_config_hash=parsing_config_hash,
        ),
        parsing_config_hash=parsing_config_hash,
        output_path=output_path,
    )


def _configuration_payload(
    *,
    config: ChunkingConfig,
    parsing_config_hash: str,
) -> dict[str, Any]:
    """Return the complete persisted cache identity."""
    if not isinstance(parsing_config_hash, str) or not parsing_config_hash:
        raise ValueError("parsing_config_hash must be a non-empty string.")

    return {
        "chunking_config": config.model_dump(mode="json"),
        "parsing_config_hash": parsing_config_hash,
    }


def _validate_and_sort_chunks(
    chunks: Iterable[Chunk],
) -> list[Chunk]:
    """Validate unique chunk IDs and return deterministic ordering."""
    chunks_by_id: dict[str, Chunk] = {}

    for chunk in chunks:
        if chunk.chunk_id in chunks_by_id:
            raise ValueError(
                f"Duplicate chunk_id in chunking result: {chunk.chunk_id!r}."
            )

        chunks_by_id[chunk.chunk_id] = chunk

    return sorted(
        chunks_by_id.values(),
        key=lambda chunk: chunk.chunk_id,
    )


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one JSON object atomically."""
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
    """Write JSON Lines records atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    temporary_path.replace(path)
