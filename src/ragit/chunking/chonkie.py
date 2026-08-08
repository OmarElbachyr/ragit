"""Chonkie adapter for the RAGit chunking contract."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ragit.chunking.base import BaseChunker, ChunkSpan
from ragit.chunking.models import ChunkingConfig
from ragit.chunking.registry import register_chunker


_SUPPORTED_CHUNKERS = {
    "token",
    "recursive",
    "sentence",
}

_ALLOWED_OPTIONS = {
    "token": {
        "tokenizer",
        "chunk_size",
        "chunk_overlap",
    },
    "recursive": {
        "tokenizer",
        "chunk_size",
        "min_characters_per_chunk",
    },
    "sentence": {
        "tokenizer",
        "chunk_size",
        "chunk_overlap",
        "min_sentences_per_chunk",
        "min_characters_per_sentence",
    },
}

_DEFAULTS: dict[str, dict[str, Any]] = {
    "token": {
        "tokenizer": "character",
        "chunk_size": 2048,
        "chunk_overlap": 0,
    },
    "recursive": {
        "tokenizer": "character",
        "chunk_size": 2048,
        "min_characters_per_chunk": 24,
    },
    "sentence": {
        "tokenizer": "character",
        "chunk_size": 2048,
        "chunk_overlap": 0,
        "min_sentences_per_chunk": 1,
        "min_characters_per_sentence": 12,
    },
}


class ChonkieChunker(BaseChunker):
    """Use selected Chonkie strategies behind the RAGit chunking contract."""

    chunker_names = frozenset(_SUPPORTED_CHUNKERS)

    @classmethod
    def validate_config(cls, config: ChunkingConfig) -> None:
        """Validate the selected Chonkie strategy and its public options."""
        super().validate_config(config)

        if config.chunker_name not in _SUPPORTED_CHUNKERS:
            raise ValueError(
                f"Unsupported Chonkie chunker {config.chunker_name!r}. "
                f"Supported chunkers: {sorted(_SUPPORTED_CHUNKERS)}."
            )

        allowed_options = _ALLOWED_OPTIONS[config.chunker_name]
        unsupported_options = set(config.options) - allowed_options

        if unsupported_options:
            raise ValueError(
                f"Unsupported options for Chonkie "
                f"{config.chunker_name!r}: {sorted(unsupported_options)}. "
                f"Supported options: {sorted(allowed_options)}."
            )

        effective_options = {
            **_DEFAULTS[config.chunker_name],
            **config.options,
        }

        _validate_common_options(effective_options)

        if config.chunker_name == "token":
            _validate_token_options(effective_options)
        elif config.chunker_name == "recursive":
            _validate_recursive_options(effective_options)
        elif config.chunker_name == "sentence":
            _validate_sentence_options(effective_options)

    def _chunk_text(
        self,
        *,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        """Run Chonkie and normalize its native chunks into exact spans."""
        self.validate_config(config)

        chunker_class = _load_chonkie_chunker_class(config.chunker_name)
        effective_options = {
            **_DEFAULTS[config.chunker_name],
            **config.options,
        }
        chunker = chunker_class(**effective_options)
        native_chunks = chunker.chunk(text)
        chonkie_version = _get_chonkie_version()

        spans: list[ChunkSpan] = []

        for native_chunk in native_chunks:
            content = getattr(native_chunk, "text", None)
            start_index = getattr(native_chunk, "start_index", None)
            end_index = getattr(native_chunk, "end_index", None)

            if not isinstance(content, str):
                raise ValueError("Chonkie returned a chunk without string text.")

            if not isinstance(start_index, int) or isinstance(start_index, bool):
                raise ValueError(
                    "Chonkie returned a chunk without an integer start_index."
                )

            if not isinstance(end_index, int) or isinstance(end_index, bool):
                raise ValueError(
                    "Chonkie returned a chunk without an integer end_index."
                )

            metadata: dict[str, Any] = {
                "backend": "chonkie",
                "chunker_name": config.chunker_name,
                "chunker_version": chonkie_version,
            }

            token_count = getattr(native_chunk, "token_count", None)
            if isinstance(token_count, int) and not isinstance(token_count, bool):
                metadata["token_count"] = token_count

            spans.append(
                ChunkSpan(
                    content=content,
                    start_index=start_index,
                    end_index=end_index,
                    metadata=metadata,
                )
            )

        return spans


def _load_chonkie_chunker_class(chunker_name: str) -> type[Any]:
    """Import only the Chonkie class needed by the selected strategy."""
    try:
        from chonkie import RecursiveChunker, SentenceChunker, TokenChunker
    except ImportError as error:
        raise ImportError(
            "Chonkie is required for Chonkie-based chunking. "
            "Install the project dependencies containing 'chonkie'."
        ) from error

    chunker_classes = {
        "token": TokenChunker,
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
    }

    return chunker_classes[chunker_name]


def _validate_common_options(options: dict[str, Any]) -> None:
    """Validate options shared by the initial Chonkie chunkers."""
    tokenizer = options.get("tokenizer")
    chunk_size = options.get("chunk_size")

    if not isinstance(tokenizer, str) or not tokenizer.strip():
        raise ValueError(
            "The 'tokenizer' option must currently be a non-empty string."
        )

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("The 'chunk_size' option must be an integer.")

    if chunk_size <= 0:
        raise ValueError("The 'chunk_size' option must be greater than zero.")


def _validate_token_options(options: dict[str, Any]) -> None:
    """Validate TokenChunker-specific options."""
    chunk_overlap = options["chunk_overlap"]

    if not isinstance(chunk_overlap, (int, float)) or isinstance(
        chunk_overlap,
        bool,
    ):
        raise ValueError(
            "The 'chunk_overlap' option must be an integer or float."
        )

    if chunk_overlap < 0:
        raise ValueError("The 'chunk_overlap' option must be non-negative.")


def _validate_recursive_options(options: dict[str, Any]) -> None:
    """Validate RecursiveChunker-specific options."""
    minimum = options["min_characters_per_chunk"]

    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise ValueError(
            "The 'min_characters_per_chunk' option must be an integer."
        )

    if minimum <= 0:
        raise ValueError(
            "The 'min_characters_per_chunk' option must be greater than zero."
        )


def _validate_sentence_options(options: dict[str, Any]) -> None:
    """Validate SentenceChunker-specific options."""
    chunk_overlap = options["chunk_overlap"]
    min_sentences = options["min_sentences_per_chunk"]
    min_characters = options["min_characters_per_sentence"]

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValueError("The 'chunk_overlap' option must be an integer.")

    if chunk_overlap < 0:
        raise ValueError("The 'chunk_overlap' option must be non-negative.")

    if not isinstance(min_sentences, int) or isinstance(min_sentences, bool):
        raise ValueError(
            "The 'min_sentences_per_chunk' option must be an integer."
        )

    if min_sentences <= 0:
        raise ValueError(
            "The 'min_sentences_per_chunk' option must be greater than zero."
        )

    if not isinstance(min_characters, int) or isinstance(min_characters, bool):
        raise ValueError(
            "The 'min_characters_per_sentence' option must be an integer."
        )

    if min_characters <= 0:
        raise ValueError(
            "The 'min_characters_per_sentence' option must be greater than zero."
        )


def _get_chonkie_version() -> str | None:
    """Return the installed Chonkie package version."""
    try:
        return version("chonkie")
    except PackageNotFoundError:
        return None


register_chunker(ChonkieChunker)
