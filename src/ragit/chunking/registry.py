"""Registry for RAGit chunker implementations."""

from __future__ import annotations

from collections.abc import Iterable

from ragit.chunking.base import BaseChunker


class ChunkerRegistryError(ValueError):
    """Raised for invalid chunker registration or lookup."""


_CHUNKERS: dict[str, type[BaseChunker]] = {}


def register_chunker(
    chunker_class: type[BaseChunker],
    names: Iterable[str] | None = None,
) -> type[BaseChunker]:
    """Register one chunker class under one or more public names.

    When ``names`` is omitted, the class must expose ``chunker_names``.
    This is intentionally a small registry, not a plugin framework.
    """
    if not isinstance(chunker_class, type) or not issubclass(
        chunker_class,
        BaseChunker,
    ):
        raise ChunkerRegistryError(
            "Registered chunkers must be BaseChunker subclasses."
        )

    if names is None:
        declared_names = getattr(chunker_class, "chunker_names", None)
        declared_name = getattr(chunker_class, "chunker_name", None)

        if declared_names is not None:
            names = declared_names
        elif declared_name is not None:
            names = [declared_name]

    if names is None:
        raise ChunkerRegistryError(
            f"Chunker {chunker_class.__name__!r} must declare "
            "chunker_name or chunker_names, or names must be supplied "
            "to register_chunker()."
        )

    normalized_names = []

    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ChunkerRegistryError(
                "Chunker registry names must be non-empty strings."
            )
        normalized_names.append(name.strip())

    if not normalized_names:
        raise ChunkerRegistryError(
            "At least one chunker registry name is required."
        )

    duplicates = [
        name
        for name in normalized_names
        if name in _CHUNKERS and _CHUNKERS[name] is not chunker_class
    ]

    if duplicates:
        raise ChunkerRegistryError(
            f"Chunker names already registered: {sorted(duplicates)}."
        )

    for name in normalized_names:
        _CHUNKERS[name] = chunker_class

    return chunker_class


def get_chunker(chunker_name: str) -> type[BaseChunker]:
    """Return the registered chunker class for one public name."""
    if not isinstance(chunker_name, str) or not chunker_name.strip():
        raise ChunkerRegistryError(
            "chunker_name must be a non-empty string."
        )

    normalized_name = chunker_name.strip()

    try:
        return _CHUNKERS[normalized_name]
    except KeyError as error:
        registered = sorted(_CHUNKERS)
        registered_text = ", ".join(registered) if registered else "none"
        raise ChunkerRegistryError(
            f"Unknown chunker {normalized_name!r}. "
            f"Registered chunkers: {registered_text}."
        ) from error
