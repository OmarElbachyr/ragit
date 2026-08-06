"""Tests for parsing contracts, registry, and persistence."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragit.data.models import Document, Page
from ragit.parsing import (
    BaseParser,
    PageStatus,
    ParsedPage,
    ParserRegistryError,
    ParsingConfig,
    ParsingConfigurationError,
    configuration_hash,
    get_parser,
    load_parsing_result,
    register_parser,
    save_parsing_result,
    validate_same_output_format,
)
from ragit.parsing.registry import clear_registry


class ExampleParser(BaseParser):
    parser_name = "example"
    supported_output_formats = frozenset({"text"})

    def parse_document(
        self,
        document: Document,
        source_pages: Sequence[Page],
        config: ParsingConfig,
    ) -> list[ParsedPage]:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def test_valid_page_states() -> None:
    successful = ParsedPage(
        page_id=0,
        doc_id="report",
        page_number=0,
        content="Parsed text",
        content_format="text",
        status=PageStatus.SUCCESS,
    )

    empty = ParsedPage(
        page_id=1,
        doc_id="report",
        page_number=1,
        content="",
        content_format="text",
        status=PageStatus.EMPTY,
    )

    failed = ParsedPage(
        page_id=2,
        doc_id="report",
        page_number=2,
        content=None,
        content_format="text",
        status=PageStatus.FAILED,
        metadata={
            "error_type": "ParsingError",
            "error_message": "Could not parse page.",
        },
    )

    assert successful.content == "Parsed text"
    assert empty.content == ""
    assert failed.content is None


@pytest.mark.parametrize(
    ("status", "content"),
    [
        ("success", ""),
        ("success", None),
        ("empty", "content"),
        ("empty", None),
        ("failed", ""),
        ("failed", "content"),
    ],
)
def test_rejects_inconsistent_status_and_content(
    status: str,
    content: str | None,
) -> None:
    with pytest.raises(ValidationError):
        ParsedPage(
            page_id=0,
            doc_id="report",
            page_number=0,
            content=content,
            content_format="text",
            status=status,
        )


def test_configuration_hash_is_deterministic() -> None:
    first = ParsingConfig(
        parser_name="example",
        output_format="markdown",
        options={
            "ocr": True,
            "layout": {
                "tables": True,
                "images": False,
            },
        },
    )

    second = ParsingConfig(
        parser_name="example",
        output_format="markdown",
        options={
            "layout": {
                "images": False,
                "tables": True,
            },
            "ocr": True,
        },
    )

    assert configuration_hash(first) == configuration_hash(second)


def test_saves_and_reloads_all_page_states(
    tmp_path: Path,
) -> None:
    pdfs_path = tmp_path / "pdfs"
    pdfs_path.mkdir()

    config = ParsingConfig(
        parser_name="example",
        output_format="markdown",
        options={"tables": True},
    )

    pages = [
        ParsedPage(
            page_id=10,
            doc_id="report",
            page_number=0,
            content="# Content",
            content_format="markdown",
            status="success",
            metadata={
                "parser_name": "example",
                "parser_version": "1.0",
            },
        ),
        ParsedPage(
            page_id=11,
            doc_id="report",
            page_number=1,
            content="",
            content_format="markdown",
            status="empty",
            metadata={"parser_name": "example"},
        ),
        ParsedPage(
            page_id=12,
            doc_id="report",
            page_number=2,
            content=None,
            content_format="markdown",
            status="failed",
            metadata={
                "parser_name": "example",
                "error_type": "ParsingError",
                "error_message": "Failure details",
            },
        ),
    ]

    saved = save_parsing_result(
        pdfs_path=pdfs_path,
        config=config,
        pages=pages,
    )

    loaded = load_parsing_result(
        pdfs_path=pdfs_path,
        parser_name=config.parser_name,
        config_hash=saved.config_hash,
    )

    assert loaded.config == config
    assert loaded.pages == pages
    assert loaded.pages[2].status == PageStatus.FAILED
    assert loaded.pages[2].metadata["error_message"] == "Failure details"


def test_parser_validates_requested_output_format() -> None:
    supported = ParsingConfig(
        parser_name="example",
        output_format="text",
    )
    ExampleParser.validate_config(supported)

    unsupported = ParsingConfig(
        parser_name="example",
        output_format="markdown",
    )

    with pytest.raises(
        ParsingConfigurationError,
        match="does not support output format",
    ):
        ExampleParser.validate_config(unsupported)


def test_rejects_mixed_comparison_formats() -> None:
    configs = [
        ParsingConfig(
            parser_name="first",
            output_format="text",
        ),
        ParsingConfig(
            parser_name="second",
            output_format="markdown",
        ),
    ]

    with pytest.raises(
        ParsingConfigurationError,
        match="same output format",
    ):
        validate_same_output_format(configs)


def test_registry_registers_and_resolves_parser() -> None:
    register_parser(ExampleParser)

    assert get_parser("example") is ExampleParser

    with pytest.raises(
        ParserRegistryError,
        match="already registered",
    ):
        register_parser(ExampleParser)

    with pytest.raises(
        ParserRegistryError,
        match="Unknown parser",
    ):
        get_parser("missing")