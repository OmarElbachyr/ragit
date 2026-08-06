"""Persistence for normalized parsing outputs."""

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragit.parsing.models import (
    ParsedPage,
    ParsingConfig,
    ParsingResult,
    ParsingStorageError,
    ParsingValidationError,
)


CONFIG_FILE = "config.json"
PAGES_FILE = "pages.jsonl"


def configuration_hash(config: ParsingConfig) -> str:
    """Return a deterministic SHA-256 hash for a parser configuration."""
    serialized = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def parsing_output_path(
    pdfs_path: str | Path,
    config: ParsingConfig,
) -> Path:
    """Return the canonical directory for a parsing configuration."""
    pdfs_path = Path(pdfs_path)

    return (
        pdfs_path
        / ".ragit"
        / "parsing"
        / config.parser_name
        / configuration_hash(config)
    )


def save_parsing_result(
    pdfs_path: str | Path,
    config: ParsingConfig,
    pages: Iterable[ParsedPage],
    *,
    replace: bool = False,
) -> ParsingResult:
    """Persist a normalized parsing result.

    Existing parsing output is not replaced unless ``replace=True``.
    """
    pages = list(pages)
    validate_parsed_pages(pages, config)

    output_path = parsing_output_path(pdfs_path, config)
    output_parent = output_path.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not replace:
        raise ParsingStorageError(
            f"Parsing output already exists: {output_path}. "
            "Pass replace=True to overwrite it."
        )

    temporary_path = output_parent / (
        f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    )

    try:
        temporary_path.mkdir(parents=False)

        config_path = temporary_path / CONFIG_FILE
        pages_path = temporary_path / PAGES_FILE

        with config_path.open("w", encoding="utf-8") as file:
            json.dump(
                config.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            file.write("\n")

        with pages_path.open("w", encoding="utf-8") as file:
            for page in pages:
                file.write(
                    json.dumps(
                        page.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if output_path.exists():
            shutil.rmtree(output_path)

        temporary_path.replace(output_path)

    except Exception as error:
        shutil.rmtree(temporary_path, ignore_errors=True)

        if isinstance(error, ParsingStorageError):
            raise

        raise ParsingStorageError(
            f"Could not save parsing result to {output_path}: {error}"
        ) from error

    return ParsingResult(
        config=config,
        config_hash=configuration_hash(config),
        output_path=output_path,
        pages=pages,
    )


def load_parsing_result(
    pdfs_path: str | Path,
    parser_name: str,
    config_hash: str,
) -> ParsingResult:
    """Load and validate a persisted parsing result."""
    output_path = (
        Path(pdfs_path)
        / ".ragit"
        / "parsing"
        / parser_name
        / config_hash
    )

    config_path = output_path / CONFIG_FILE
    pages_path = output_path / PAGES_FILE

    if not config_path.exists():
        raise ParsingStorageError(
            f"Missing parsing configuration: {config_path}"
        )

    if not pages_path.exists():
        raise ParsingStorageError(
            f"Missing parsed pages file: {pages_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config_record = json.load(file)

        config = ParsingConfig.model_validate(config_record)

    except (json.JSONDecodeError, ValidationError) as error:
        raise ParsingStorageError(
            f"Invalid parsing configuration in {config_path}: {error}"
        ) from error

    if config.parser_name != parser_name:
        raise ParsingStorageError(
            f"Stored parser name {config.parser_name!r} does not match "
            f"directory parser name {parser_name!r}."
        )

    actual_hash = configuration_hash(config)

    if actual_hash != config_hash:
        raise ParsingStorageError(
            f"Stored configuration hash is {actual_hash}, "
            f"but directory hash is {config_hash}."
        )

    pages = read_parsed_pages(pages_path)
    validate_parsed_pages(pages, config)

    return ParsingResult(
        config=config,
        config_hash=config_hash,
        output_path=output_path,
        pages=pages,
    )


def read_parsed_pages(path: Path) -> list[ParsedPage]:
    """Read normalized pages from JSON Lines."""
    pages = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record: Any = json.loads(line)
                pages.append(ParsedPage.model_validate(record))
            except (json.JSONDecodeError, ValidationError) as error:
                raise ParsingStorageError(
                    f"Invalid parsed page in {path} at line "
                    f"{line_number}: {error}"
                ) from error

    return pages


def validate_parsed_pages(
    pages: list[ParsedPage],
    config: ParsingConfig,
) -> None:
    """Perform basic cross-record validation."""
    page_ids = [page.page_id for page in pages]

    if len(page_ids) != len(set(page_ids)):
        raise ParsingValidationError(
            "Parsed page IDs must be unique."
        )

    page_locations = [
        (page.doc_id, page.page_number)
        for page in pages
    ]

    if len(page_locations) != len(set(page_locations)):
        raise ParsingValidationError(
            "Each document page may appear only once."
        )

    mismatched_formats = [
        page.page_id
        for page in pages
        if page.content_format != config.output_format
    ]

    if mismatched_formats:
        raise ParsingValidationError(
            "Parsed page content formats must match the parser "
            f"configuration. Mismatched page IDs: {mismatched_formats}."
        )