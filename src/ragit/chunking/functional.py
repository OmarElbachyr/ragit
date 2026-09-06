"""Low-boilerplate function-based custom chunkers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

from ragit.chunking.base import BaseChunker, ChunkSpan
from ragit.chunking.models import ChunkingConfig
from ragit.chunking.registry import register_chunker


ChunkOutput: TypeAlias = str | ChunkSpan | Mapping[str, Any]
ChunkFunction: TypeAlias = Callable[
    [str, dict[str, Any]],
    Iterable[ChunkOutput],
]


class FunctionChunker(BaseChunker):
    """Adapt a user function to RAGit's document chunking lifecycle."""

    chunker_name = ""
    _chunk_function: ChunkFunction

    def _chunk_text(
        self,
        *,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        outputs = self._chunk_function(text, dict(config.options))
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Iterable):
            raise TypeError(
                "A function chunker must return an iterable of strings, "
                "ChunkSpan objects, or span mappings."
            )

        spans: list[ChunkSpan] = []
        search_from = 0
        for position, output in enumerate(outputs):
            span = _normalize_output(
                output,
                text=text,
                search_from=search_from,
                position=position,
            )
            spans.append(span)
            search_from = span.end_index
        return spans


def chunker(name: str) -> Callable[[ChunkFunction], ChunkFunction]:
    """Register a text function as a named RAGit chunker."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A function chunker name must be a non-empty string.")
    normalized_name = name.strip()

    def decorator(function: ChunkFunction) -> ChunkFunction:
        if not callable(function):
            raise TypeError("@chunker can only decorate a callable.")

        class_name = "".join(
            part.capitalize()
            for part in normalized_name.replace("-", "_").split("_")
        )
        chunker_class = type(
            f"{class_name or 'Custom'}FunctionChunker",
            (FunctionChunker,),
            {
                "__module__": function.__module__,
                "chunker_name": normalized_name,
                "_chunk_function": staticmethod(function),
            },
        )
        register_chunker(chunker_class)
        return function

    return decorator


def _normalize_output(
    output: ChunkOutput,
    *,
    text: str,
    search_from: int,
    position: int,
) -> ChunkSpan:
    if isinstance(output, str):
        if not output:
            raise ValueError(f"Function chunk {position} is empty.")
        start_index = text.find(output, search_from)
        if start_index < 0:
            raise ValueError(
                f"Function chunk {position} could not be mapped to the source "
                "text after the preceding chunk. Return an explicit ChunkSpan "
                "when chunks overlap or are reordered."
            )
        return ChunkSpan(
            content=output,
            start_index=start_index,
            end_index=start_index + len(output),
        )

    if isinstance(output, ChunkSpan):
        span = output
    elif isinstance(output, Mapping):
        try:
            span = ChunkSpan(**dict(output))
        except TypeError as error:
            raise TypeError(
                f"Function chunk {position} is not a valid span mapping."
            ) from error
    else:
        raise TypeError(
            f"Function chunk {position} must be a string, ChunkSpan, or mapping."
        )

    if text[span.start_index : span.end_index] != span.content:
        raise ValueError(
            f"Function chunk {position} content does not match its source span."
        )
    return span
