"""Document metadata creation and loading."""

import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from ragit.data.models import Document
from ragit.data.validation import CollectionValidationError


def create_default_metadata(
    pdf_files: list[Path],
    page_counts: dict[str, int],
) -> list[Document]:
    """Create the minimum metadata required by a ragit collection."""
    return [
        Document(
            file_name=pdf_path.name,
            doc_id=pdf_path.stem,
            local_path=str(pdf_path),
            page_number=page_counts[pdf_path.stem],
        )
        for pdf_path in pdf_files
    ]


def extract_optional_metadata(
    pdf_files: list[Path],
    documents: list[Document],
) -> list[Document]:
    """Extract optional document metadata.

    This pipeline will later detect fields such as:

    - doc_language
    - doc_type
    - visual_types
    """
    raise NotImplementedError(
        "Optional metadata extraction is not implemented yet."
    )


def load_metadata(path: Path) -> list[Document]:
    """Load metadata while preserving optional additional fields."""
    documents = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                documents.append(Document.model_validate(record))
            except (json.JSONDecodeError, ValidationError) as error:
                raise CollectionValidationError(
                    f"Invalid metadata in {path} at line "
                    f"{line_number}: {error}"
                ) from error

    return documents


def write_metadata(
    path: Path,
    documents: Iterable[Document],
) -> None:
    """Write document metadata atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(
                    document.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(path)