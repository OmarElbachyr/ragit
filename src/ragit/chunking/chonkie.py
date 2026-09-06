"""Chonkie adapter for the RAGit chunking contract."""

from __future__ import annotations

import json
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect
from typing import Any

from ragit.chunking.base import BaseChunker, ChunkSpan
from ragit.chunking.models import ChunkingConfig
from ragit.chunking.registry import register_chunker


_SUPPORTED_CHUNKERS = {
    "fast",
    "late",
    "neural",
    "token",
    "recursive",
    "semantic",
    "sentence",
}

_CHONKIE_DOC_URLS = {
    name: f"https://docs.chonkie.ai/oss/chunkers/{name}-chunker"
    for name in _SUPPORTED_CHUNKERS
}

_CHUNKER_DESCRIPTIONS = {
    "page": "Create one chunk for each usable PDF page.",
    "token": "Split text into fixed-size token chunks.",
    "sentence": "Split text while preserving sentence boundaries.",
    "recursive": "Split structured text recursively into smaller chunks.",
    "fast": "Split text at high speed using byte-size boundaries.",
    "semantic": "Group text using semantic similarity.",
    "late": "Create chunks with document-context-aware embeddings.",
    "neural": "Detect semantic shifts using a neural model.",
}


class ChunkerConfigurationError(ValueError):
    """Raised when Chonkie cannot construct a requested chunker."""


class ChonkieChunker(BaseChunker):
    """Use selected Chonkie strategies behind the RAGit chunking contract."""

    chunker_names = frozenset(_SUPPORTED_CHUNKERS)

    @classmethod
    def validate_config(cls, config: ChunkingConfig) -> None:
        """Validate the strategy and RAGit persistence requirements."""
        super().validate_config(config)

        if config.chunker_name not in _SUPPORTED_CHUNKERS:
            raise ValueError(
                f"Unsupported Chonkie chunker {config.chunker_name!r}. "
                f"Supported chunkers: {sorted(_SUPPORTED_CHUNKERS)}."
            )


    def _chunk_text(
        self,
        *,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        """Run Chonkie and normalize its native chunks into exact spans."""
        self.validate_config(config)

        chunker_class = _load_chonkie_chunker_class(config.chunker_name)
        try:
            chunker = chunker_class(**config.options)
        except (TypeError, ValueError) as error:
            documentation = _CHONKIE_DOC_URLS[config.chunker_name]
            raise ChunkerConfigurationError(
                f"Invalid options for chunker {config.chunker_name!r}: {error}. "
                f"Inspect them with describe_chunker({config.chunker_name!r}) "
                f"or see {documentation}."
            ) from error
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

            if config.chunker_name == "late":
                embedding = getattr(native_chunk, "embedding", None)
                if embedding is None:
                    raise ValueError(
                        "Chonkie LateChunker returned a chunk without an embedding."
                    )
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                if not isinstance(embedding, list) or not embedding:
                    raise ValueError(
                        "Chonkie LateChunker returned an invalid embedding."
                    )
                try:
                    metadata["embedding"] = [float(value) for value in embedding]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Chonkie LateChunker returned a non-numeric embedding."
                    ) from error
                embedding_model = config.options.get("embedding_model")
                if isinstance(embedding_model, str):
                    metadata["embedding_model"] = embedding_model

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
        chonkie = import_module("chonkie")
    except ImportError as error:
        raise ImportError(
            "Chonkie is required for Chonkie-based chunking. "
            "Install the project dependencies containing 'chonkie'."
        ) from error

    class_names = {
        "fast": "FastChunker",
        "late": "LateChunker",
        "neural": "NeuralChunker",
        "token": "TokenChunker",
        "recursive": "RecursiveChunker",
        "semantic": "SemanticChunker",
        "sentence": "SentenceChunker",
    }
    class_name = class_names[chunker_name]
    chunker_class = getattr(chonkie, class_name, None)
    if chunker_class is None:
        raise ImportError(
            f"Installed Chonkie does not provide {class_name}. Upgrade Chonkie "
            "and install the optional dependencies required by this strategy."
        )
    return chunker_class


def describe_chunker(chunker_name: str) -> dict[str, Any]:
    """Describe a supported chunker and its runtime constructor options."""
    if chunker_name == "page":
        return {
            "name": "page",
            "class": "PageChunker",
            "backend": "ragit",
            "description": _CHUNKER_DESCRIPTIONS["page"],
            "options": {},
            "accepts_additional_options": False,
            "documentation": "docs/chunking.md#page",
        }

    if chunker_name not in _SUPPORTED_CHUNKERS:
        supported = sorted({"page", *_SUPPORTED_CHUNKERS})
        raise ValueError(
            f"Unknown chunker {chunker_name!r}. Supported chunkers: {supported}."
        )

    chunker_class = _load_chonkie_chunker_class(chunker_name)
    try:
        signature = inspect.signature(chunker_class)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Could not inspect Chonkie chunker {chunker_name!r}."
        ) from error

    options: dict[str, dict[str, Any]] = {}
    accepts_additional_options = False
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_additional_options = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue

        option: dict[str, Any] = {
            "required": parameter.default is inspect.Parameter.empty,
        }
        if parameter.annotation is not inspect.Parameter.empty:
            option["type"] = inspect.formatannotation(parameter.annotation)
        if parameter.default is not inspect.Parameter.empty:
            option["default"] = _displayable_default(parameter.default)
        options[parameter.name] = option

    return {
        "name": chunker_name,
        "class": chunker_class.__name__,
        "backend": "chonkie",
        "description": _CHUNKER_DESCRIPTIONS[chunker_name],
        "options": options,
        "accepts_additional_options": accepts_additional_options,
        "documentation": _CHONKIE_DOC_URLS[chunker_name],
    }


def list_chunker_options(chunker_name: str) -> dict[str, dict[str, Any]]:
    """Return the discoverable constructor options for one chunker."""
    return describe_chunker(chunker_name)["options"]


def _displayable_default(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _get_chonkie_version() -> str | None:
    """Return the installed Chonkie package version."""
    try:
        return version("chonkie")
    except PackageNotFoundError:
        return None


register_chunker(ChonkieChunker)
