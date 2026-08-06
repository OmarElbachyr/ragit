"""Registration and lookup of PDF parser adapters."""

from collections.abc import Iterable

from ragit.parsing.base import BaseParser
from ragit.parsing.models import (
    ContentFormat,
    ParsingConfig,
    ParsingConfigurationError,
    ParsingError,
)


class ParserRegistryError(ParsingError):
    """Raised when parser registration or resolution fails."""


_PARSERS: dict[str, type[BaseParser]] = {}


def register_parser(
    parser_class: type[BaseParser],
    *,
    replace: bool = False,
) -> type[BaseParser]:
    """Register a parser implementation by its stable name."""
    if not isinstance(parser_class, type) or not issubclass(
        parser_class,
        BaseParser,
    ):
        raise ParserRegistryError(
            "Registered parsers must subclass BaseParser."
        )

    parser_name = getattr(parser_class, "parser_name", "").strip()

    if not parser_name:
        raise ParserRegistryError(
            "Registered parsers must define a non-empty parser_name."
        )

    supported_formats = getattr(
        parser_class,
        "supported_output_formats",
        frozenset(),
    )

    if not supported_formats:
        raise ParserRegistryError(
            f"Parser {parser_name!r} must support at least one output format."
        )

    invalid_formats = set(supported_formats) - {"text", "markdown"}

    if invalid_formats:
        raise ParserRegistryError(
            f"Parser {parser_name!r} declares unsupported formats: "
            f"{sorted(invalid_formats)}."
        )

    if parser_name in _PARSERS and not replace:
        raise ParserRegistryError(
            f"Parser {parser_name!r} is already registered."
        )

    _PARSERS[parser_name] = parser_class
    return parser_class


def get_parser(parser_name: str) -> type[BaseParser]:
    """Resolve a registered parser class by name."""
    try:
        return _PARSERS[parser_name]
    except KeyError as error:
        available = ", ".join(sorted(_PARSERS)) or "none"

        raise ParserRegistryError(
            f"Unknown parser {parser_name!r}. "
            f"Registered parsers: {available}."
        ) from error


def validate_same_output_format(
    configs: Iterable[ParsingConfig],
) -> ContentFormat:
    """Require all parser configurations to use one output format."""
    configs = list(configs)

    if not configs:
        raise ParsingConfigurationError(
            "At least one parser configuration is required."
        )

    formats = {config.output_format for config in configs}

    if len(formats) != 1:
        raise ParsingConfigurationError(
            "Parser comparisons must use the same output format. "
            f"Received: {sorted(formats)}."
        )

    return configs[0].output_format


def clear_registry() -> None:
    """Clear the parser registry.

    Intended primarily for isolated tests.
    """
    _PARSERS.clear()