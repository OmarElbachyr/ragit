"""Public parsing contracts and persistence API."""

from ragit.parsing.base import BaseParser
from ragit.parsing.models import (
    ContentFormat,
    PageStatus,
    ParsedPage,
    ParsingConfig,
    ParsingConfigurationError,
    ParsingError,
    ParsingResult,
    ParsingStorageError,
    ParsingValidationError,
)
from ragit.parsing.parsers import (
    DoclingParser,
    PyMuPDF4LLMParser,
)
from ragit.parsing.registry import (
    ParserRegistryError,
    get_parser,
    register_parser,
    validate_same_output_format,
)
from ragit.parsing.storage import (
    configuration_hash,
    load_parsing_result,
    parsing_output_path,
    save_parsing_result,
)

__all__ = [
    "BaseParser",
    "ContentFormat",
    "DoclingParser",
    "PageStatus",
    "ParsedPage",
    "ParserRegistryError",
    "ParsingConfig",
    "ParsingConfigurationError",
    "ParsingError",
    "ParsingResult",
    "ParsingStorageError",
    "ParsingValidationError",
    "PyMuPDF4LLMParser",
    "configuration_hash",
    "get_parser",
    "load_parsing_result",
    "parsing_output_path",
    "register_parser",
    "save_parsing_result",
    "validate_same_output_format",
]