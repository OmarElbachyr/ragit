"""Tests for the Docling parser adapter."""

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
from ragit.parsing.parsers.docling import DoclingParser
import ragit.parsing.parsers.docling as parser_module


class MockDoclingDocument:
    """Mock native Docling document exports."""

    def __init__(
        self,
        outputs: dict[int, str | Exception | None],
    ) -> None:
        self.outputs = outputs
        self.markdown_calls: list[tuple[int, bool]] = []
        self.text_calls: list[tuple[int, bool]] = []

    def _output(self, page_no: int):
        output = self.outputs.get(page_no)

        if isinstance(output, Exception):
            raise output

        return output

    def export_to_markdown(
        self,
        *,
        page_no: int,
        traverse_pictures: bool,
    ):
        self.markdown_calls.append(
            (page_no, traverse_pictures)
        )
        return self._output(page_no)

    def export_to_text(
        self,
        *,
        page_no: int,
        traverse_pictures: bool,
    ):
        self.text_calls.append(
            (page_no, traverse_pictures)
        )
        return self._output(page_no)


class MockConverter:
    """Mock Docling converter."""

    def __init__(
        self,
        docling_document=None,
        error: Exception | None = None,
    ) -> None:
        self.docling_document = docling_document
        self.error = error

    def convert(self, path: str):
        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            document=self.docling_document
        )


@pytest.fixture
def document(tmp_path: Path) -> Document:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.touch()

    return Document(
        file_name=pdf_path.name,
        doc_id="report",
        local_path=str(pdf_path),
        page_number=3,
        language="fr",
    )


@pytest.fixture
def source_pages() -> list[Page]:
    return [
        Page(
            page_id=40,
            doc_id="report",
            page_number=0,
        ),
        Page(
            page_id=41,
            doc_id="report",
            page_number=1,
        ),
        Page(
            page_id=42,
            doc_id="report",
            page_number=2,
        ),
    ]


def install_converter(
    monkeypatch: pytest.MonkeyPatch,
    converter: MockConverter,
) -> None:
    monkeypatch.setattr(
        parser_module,
        "_build_converter",
        lambda effective_options: converter,
    )

    monkeypatch.setattr(
        parser_module,
        "_get_docling_version",
        lambda: "1.0",
    )


def test_native_markdown_export(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: "# Page 1",
            2: "# Page 2",
            3: "# Page 3",
        }
    )

    install_converter(
        monkeypatch,
        MockConverter(docling_document),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="markdown",
    )

    pages = DoclingParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert [page.content for page in pages] == [
        "# Page 1",
        "# Page 2",
        "# Page 3",
    ]

    assert all(
        page.content_format == "markdown"
        for page in pages
    )

    assert docling_document.markdown_calls == [
        (1, False),
        (2, False),
        (3, False),
    ]


def test_native_text_export(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: "Plain text",
        }
    )

    install_converter(
        monkeypatch,
        MockConverter(docling_document),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="text",
    )

    page = DoclingParser().parse_document(
        document,
        source_pages[:1],
        config,
    )[0]

    assert page.content == "Plain text"
    assert page.content_format == "text"
    assert page.status == PageStatus.SUCCESS
    assert docling_document.text_calls == [(1, False)]


def test_zero_based_mapping_and_page_ids(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: "First",
            2: "Second",
            3: "Third",
        }
    )

    install_converter(
        monkeypatch,
        MockConverter(docling_document),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="text",
    )

    pages = DoclingParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert [page.page_number for page in pages] == [
        0,
        1,
        2,
    ]

    assert [page.page_id for page in pages] == [
        40,
        41,
        42,
    ]

    assert docling_document.text_calls == [
        (1, False),
        (2, False),
        (3, False),
    ]


def test_empty_page(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: " \n\t ",
        }
    )

    install_converter(
        monkeypatch,
        MockConverter(docling_document),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="text",
    )

    page = DoclingParser().parse_document(
        document,
        source_pages[:1],
        config,
    )[0]

    assert page.status == PageStatus.EMPTY
    assert page.content == ""


def test_missing_page_output(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: None,
        }
    )

    install_converter(
        monkeypatch,
        MockConverter(docling_document),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="text",
    )

    page = DoclingParser().parse_document(
        document,
        source_pages[:1],
        config,
    )[0]

    assert page.status == PageStatus.FAILED
    assert page.content is None
    assert (
        page.metadata["error_type"]
        == "MissingPageOutput"
    )


def test_page_failure_does_not_stop_document(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: "First",
            2: RuntimeError("Export failed"),
            3: "Third",
        }
    )

    install_converter(
        monkeypatch,
        MockConverter(docling_document),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="markdown",
    )

    pages = DoclingParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert [page.status for page in pages] == [
        PageStatus.SUCCESS,
        PageStatus.FAILED,
        PageStatus.SUCCESS,
    ]

    assert pages[1].metadata["error_type"] == (
        "RuntimeError"
    )

    assert pages[1].metadata["error_message"] == (
        "Export failed"
    )


def test_document_conversion_failure(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_converter(
        monkeypatch,
        MockConverter(
            error=RuntimeError("Conversion failed")
        ),
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="markdown",
    )

    pages = DoclingParser().parse_document(
        document,
        source_pages,
        config,
    )

    assert len(pages) == 3

    assert all(
        page.status == PageStatus.FAILED
        for page in pages
    )

    assert all(
        page.content is None
        for page in pages
    )

    assert [page.page_id for page in pages] == [
        40,
        41,
        42,
    ]


def test_use_ocr_forces_full_page_ocr(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_document = MockDoclingDocument(
        {
            1: "OCR content",
        }
    )

    captured_options = {}

    def build_converter(effective_options):
        captured_options.update(effective_options)
        return MockConverter(docling_document)

    monkeypatch.setattr(
        parser_module,
        "_build_converter",
        build_converter,
    )

    monkeypatch.setattr(
        parser_module,
        "_get_docling_version",
        lambda: "1.0",
    )

    config = ParsingConfig(
        parser_name="docling",
        output_format="text",
        options={
            "use_ocr": True,
        },
    )

    pages = DoclingParser().parse_document(
        document,
        source_pages[:1],
        config,
    )

    assert captured_options["do_ocr"] is True
    assert captured_options["ocr_mode"] == "full_page"
    assert captured_options["ocr_language"] == ["fr"]

    assert docling_document.text_calls == [
        (1, True),
    ]

    assert (
        pages[0].metadata["parser_options"]
        == captured_options
    )


def test_rejects_unsupported_option() -> None:
    config = ParsingConfig(
        parser_name="docling",
        output_format="text",
        options={
            "unknown_option": True,
        },
    )

    with pytest.raises(
        ParsingConfigurationError,
        match="Unsupported options",
    ):
        DoclingParser.validate_config(config)


def test_parser_is_registered() -> None:
    assert get_parser("docling") is DoclingParser