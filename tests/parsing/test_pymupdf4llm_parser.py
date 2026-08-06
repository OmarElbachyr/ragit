"""Tests for the PyMuPDF4LLM parser adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ragit.data.models import Document, Page
from ragit.parsing import (
    PageStatus,
    ParsingConfig,
    ParsingConfigurationError,
    get_parser,
)
from ragit.parsing.parsers.pymupdf4llm import PyMuPDF4LLMParser
import ragit.parsing.parsers.pymupdf4llm as parser_module


@pytest.fixture
def document(tmp_path: Path) -> Document:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.touch()

    return Document(
        file_name=pdf_path.name,
        doc_id="report",
        local_path=str(pdf_path),
        page_number=3,
    )


@pytest.fixture
def source_pages() -> list[Page]:
    return [
        Page(page_id=40, doc_id="report", page_number=0),
        Page(page_id=41, doc_id="report", page_number=1),
        Page(page_id=42, doc_id="report", page_number=2),
    ]


def install_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    to_markdown=None,
    to_text=None,
) -> None:
    backend = SimpleNamespace(
        version="1.0",
        to_markdown=to_markdown,
        to_text=to_text,
    )

    monkeypatch.setattr(
        parser_module,
        "_load_backend",
        lambda: backend,
    )


def test_normalizes_markdown_output(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def to_markdown(path, **options):
        calls.append((path, options))
        page_number = options["pages"][0]
        return [{"text": f"# Page {page_number}"}]

    install_backend(
        monkeypatch,
        to_markdown=to_markdown,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="markdown",
    )

    pages = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert [page.content for page in pages] == [
        "# Page 0",
        "# Page 1",
        "# Page 2",
    ]
    assert all(
        page.content_format == "markdown"
        for page in pages
    )

    assert calls[0][1]["pages"] == [0]
    assert calls[0][1]["page_chunks"] is True


def test_normalizes_plain_text_output(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def to_text(path, **options):
        return [{"text": "Plain text output"}]

    install_backend(
        monkeypatch,
        to_text=to_text,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    )

    pages = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages[:1],
        config,
    )

    assert pages[0].content == "Plain text output"
    assert pages[0].content_format == "text"
    assert pages[0].status == PageStatus.SUCCESS


def test_preserves_zero_based_mapping_and_page_ids(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_pages = []

    def to_text(path, **options):
        requested_pages.append(options["pages"][0])

        # Deliberately expose irrelevant one-based metadata.
        return [
            {
                "text": "content",
                "metadata": {"page_number": 999},
            }
        ]

    install_backend(
        monkeypatch,
        to_text=to_text,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    )

    pages = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert requested_pages == [0, 1, 2]
    assert [page.page_number for page in pages] == [0, 1, 2]
    assert [page.page_id for page in pages] == [40, 41, 42]


def test_normalizes_whitespace_as_empty(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def to_text(path, **options):
        return [{"text": " \n\t "}]

    install_backend(
        monkeypatch,
        to_text=to_text,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    )

    page = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages[:1],
        config,
    )[0]

    assert page.status == PageStatus.EMPTY
    assert page.content == ""


def test_missing_page_output_becomes_failed(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def to_text(path, **options):
        return []

    install_backend(
        monkeypatch,
        to_text=to_text,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    )

    page = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages[:1],
        config,
    )[0]

    assert page.status == PageStatus.FAILED
    assert page.content is None
    assert page.metadata["error_type"] == "MissingPageOutput"


def test_individual_page_failure_does_not_stop_document(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def to_text(path, **options):
        page_number = options["pages"][0]

        if page_number == 1:
            raise RuntimeError("Page failed")

        return [{"text": f"Page {page_number}"}]

    install_backend(
        monkeypatch,
        to_text=to_text,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    )

    pages = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert [page.status for page in pages] == [
        PageStatus.SUCCESS,
        PageStatus.FAILED,
        PageStatus.SUCCESS,
    ]
    assert pages[1].metadata["error_type"] == "RuntimeError"
    assert pages[1].metadata["error_message"] == "Page failed"


def test_whole_document_failure_creates_placeholders(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_backend():
        raise RuntimeError("Backend initialization failed")

    monkeypatch.setattr(
        parser_module,
        "_load_backend",
        fail_backend,
    )

    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    )

    pages = PyMuPDF4LLMParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert len(pages) == 3
    assert all(page.status == PageStatus.FAILED for page in pages)
    assert all(page.content is None for page in pages)
    assert [page.page_id for page in pages] == [40, 41, 42]


def test_rejects_unsupported_options() -> None:
    config = ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
        options={"invented_option": True},
    )

    with pytest.raises(
        ParsingConfigurationError,
        match="Unsupported options",
    ):
        PyMuPDF4LLMParser.validate_config(config)


def test_parser_is_registered() -> None:
    assert get_parser("pymupdf4llm") is PyMuPDF4LLMParser