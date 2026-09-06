"""Tests for the Marker parser adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import ragit.parsing.parsers.marker as parser_module
from ragit.data.models import Document, Page
from ragit.parsing import (
    PageStatus,
    ParsingConfig,
    ParsingConfigurationError,
    get_parser,
)
from ragit.parsing.parsers.marker import MarkerParser


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


class MockConverter:
    """Mock Marker converter."""

    def __init__(
        self,
        *,
        artifact_dict,
        config,
        llm_service=None,
        output=None,
        error=None,
    ):
        self.artifact_dict = artifact_dict
        self.config = config
        self.llm_service = llm_service
        self.output = output
        self.error = error
        self.paths: list[str] = []

    def __call__(self, path: str):
        self.paths.append(path)

        if self.error is not None:
            raise self.error

        return self.output


def paginated_markdown(*contents: str) -> str:
    """Build the native paginated Markdown shape Marker returns."""
    return "".join(
        f"\n\n{{{page_number}}}{parser_module._PAGE_SEPARATOR}\n\n"
        f"{content}"
        for page_number, content in enumerate(contents)
    )


def install_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    markdown: str | None = None,
    error: Exception | None = None,
):
    """Install a mock Marker backend and return its converter instances."""
    converters: list[MockConverter] = []

    def converter_factory(*, artifact_dict, config, llm_service=None):
        converter = MockConverter(
            artifact_dict=artifact_dict,
            config=config,
            llm_service=llm_service,
            output=SimpleNamespace(markdown=markdown),
            error=error,
        )
        converters.append(converter)
        return converter

    backend = SimpleNamespace(
        PdfConverter=converter_factory,
        create_model_dict=lambda: {"models": "loaded"},
        text_from_rendered=lambda rendered: (rendered.markdown, {}, {}),
    )

    monkeypatch.setattr(parser_module, "_load_backend", lambda: backend)
    monkeypatch.setattr(parser_module, "_get_marker_version", lambda: "1.0")
    return converters


def test_normalizes_paginated_markdown(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converters = install_backend(
        monkeypatch,
        markdown=paginated_markdown("# Page 1", "# Page 2", "# Page 3"),
    )

    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
    )

    pages = MarkerParser().parse_document(document, source_pages, config)

    assert [page.content for page in pages] == [
        "# Page 1",
        "# Page 2",
        "# Page 3",
    ]
    assert [page.page_id for page in pages] == [40, 41, 42]
    assert [page.page_number for page in pages] == [0, 1, 2]
    assert all(page.status == PageStatus.SUCCESS for page in pages)
    assert all(page.content_format == "markdown" for page in pages)

    assert converters[0].config == {
        "output_format": "markdown",
        "paginate_output": True,
        "page_separator": parser_module._PAGE_SEPARATOR,
        "page_range": [0, 1, 2],
        "force_ocr": False,
        "use_llm": False,
        "extract_images": False,
    }
    assert converters[0].llm_service is None
    assert converters[0].paths == [document.local_path]


def test_maps_use_ocr_to_marker_force_ocr(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converters = install_backend(
        monkeypatch,
        markdown=paginated_markdown("# Page 1"),
    )

    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
        options={"use_ocr": True},
    )

    MarkerParser().parse_document(document, source_pages[:1], config)

    assert converters[0].config["force_ocr"] is True


def test_maps_describe_pictures_to_marker_llm_descriptions(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converters = install_backend(
        monkeypatch,
        markdown=paginated_markdown("A chart of annual revenue."),
    )

    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
        options={"describe_pictures": True},
    )

    pages = MarkerParser().parse_document(
        document,
        source_pages[:1],
        config,
    )

    assert pages[0].content == "A chart of annual revenue."
    assert converters[0].config["use_llm"] is True
    assert converters[0].config["extract_images"] is False
    assert converters[0].config["openai_base_url"] == (
        "http://127.0.0.1:8000/v1"
    )
    assert converters[0].config["openai_model"] == (
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    assert converters[0].config["openai_image_format"] == "png"
    assert converters[0].config["openai_api_key"] == "EMPTY"
    assert converters[0].llm_service == (
        "marker.services.openai.OpenAIService"
    )


def test_allows_selecting_marker_picture_description_model(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RAGIT_MARKER_VLLM_MODEL",
        "organization/custom-vision-model",
    )
    monkeypatch.setenv(
        "RAGIT_MARKER_VLLM_BASE_URL",
        "http://inference.internal:9000/v1",
    )
    converters = install_backend(
        monkeypatch,
        markdown=paginated_markdown("A picture description."),
    )

    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
        options={"describe_pictures": True},
    )

    MarkerParser().parse_document(document, source_pages[:1], config)

    assert converters[0].config["openai_model"] == (
        "organization/custom-vision-model"
    )
    assert converters[0].config["openai_base_url"] == (
        "http://inference.internal:9000/v1"
    )


def test_empty_and_missing_paginated_pages(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_backend(
        monkeypatch,
        markdown=paginated_markdown("# Page 1", " \n\t "),
    )

    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
    )

    pages = MarkerParser().parse_document(document, source_pages, config)

    assert [page.status for page in pages] == [
        PageStatus.SUCCESS,
        PageStatus.EMPTY,
        PageStatus.FAILED,
    ]
    assert pages[1].content == ""
    assert pages[2].content is None
    assert pages[2].metadata["error_type"] == "MissingPageOutput"


def test_document_failure_creates_placeholders(
    document: Document,
    source_pages: list[Page],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_backend(
        monkeypatch,
        error=RuntimeError("Marker failed"),
    )

    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
    )

    pages = MarkerParser().parse_document(document, source_pages, config)

    assert all(page.status == PageStatus.FAILED for page in pages)
    assert all(page.content is None for page in pages)
    assert pages[0].metadata["error_type"] == "RuntimeError"
    assert pages[0].metadata["error_message"] == "Marker failed"


def test_requires_markdown_output() -> None:
    config = ParsingConfig(
        parser_name="marker",
        output_format="text",
    )

    with pytest.raises(
        ParsingConfigurationError,
        match="does not support output format",
    ):
        MarkerParser.validate_config(config)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"use_ocr": "yes"}, "must be a Boolean"),
        ({"describe_pictures": "yes"}, "must be a Boolean"),
        ({"use_llm": True}, "Unsupported options"),
    ],
)
def test_validates_marker_options(
    options: dict[str, object],
    message: str,
) -> None:
    config = ParsingConfig(
        parser_name="marker",
        output_format="markdown",
        options=options,
    )

    with pytest.raises(ParsingConfigurationError, match=message):
        MarkerParser.validate_config(config)


def test_registers_marker_parser() -> None:
    assert get_parser("marker") is MarkerParser
