"""Focused tests for document-level chunking and provenance."""

from dataclasses import dataclass

import pytest

from ragit.chunking.base import (
    BaseChunker,
    ChunkSpan,
    _reconstruct_document,
)
from ragit.chunking.chonkie import ChonkieChunker
from ragit.chunking.models import Chunk, ChunkingConfig, ChunkingResult
from ragit.chunking.storage import (
    chunking_configuration_hash,
    load_chunking_result,
    save_chunking_result,
)
from ragit.parsing.models import PageStatus, ParsedPage


def _parsed_page(
    *,
    page_id: int,
    page_number: int,
    content: str | None,
    status: PageStatus,
) -> ParsedPage:
    return ParsedPage(
        page_id=page_id,
        doc_id="doc",
        page_number=page_number,
        content=content,
        content_format="text",
        status=status,
        metadata={},
    )


class StaticSpanChunker(BaseChunker):
    """Test chunker returning predetermined document-relative spans."""

    def __init__(self, spans: list[ChunkSpan]) -> None:
        self.spans = spans

    def _chunk_text(
        self,
        *,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        return self.spans


def test_reconstructs_document_and_tracks_page_ranges() -> None:
    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        ),
        _parsed_page(
            page_id=11,
            page_number=1,
            content="defg",
            status=PageStatus.SUCCESS,
        ),
    ]

    text, ranges = _reconstruct_document(pages)

    assert text == "abc\ndefg"
    assert [
        (item.page_id, item.start_index, item.end_index)
        for item in ranges
    ] == [
        (10, 0, 3),
        (11, 4, 8),
    ]


def test_chunk_provenance_can_cover_one_or_multiple_pages() -> None:
    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        ),
        _parsed_page(
            page_id=11,
            page_number=1,
            content="defg",
            status=PageStatus.SUCCESS,
        ),
    ]
    chunker = StaticSpanChunker(
        [
            ChunkSpan(
                content="ab",
                start_index=0,
                end_index=2,
            ),
            ChunkSpan(
                content="c\nd",
                start_index=2,
                end_index=5,
            ),
        ]
    )

    chunks = chunker.chunk_document(
        pages,
        ChunkingConfig(chunker_name="custom"),
    )

    assert chunks[0].page_ids == [10]
    assert chunks[1].page_ids == [10, 11]
    assert chunks[0].chunk_id == "doc:00000000"
    assert chunks[1].chunk_id == "doc:00000001"


def test_empty_and_failed_pages_add_no_text_without_renumbering() -> None:
    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        ),
        _parsed_page(
            page_id=11,
            page_number=1,
            content="",
            status=PageStatus.EMPTY,
        ),
        _parsed_page(
            page_id=12,
            page_number=2,
            content=None,
            status=PageStatus.FAILED,
        ),
        _parsed_page(
            page_id=13,
            page_number=3,
            content="xyz",
            status=PageStatus.SUCCESS,
        ),
    ]

    text, ranges = _reconstruct_document(pages)

    assert text == "abc\nxyz"
    assert [item.page_id for item in ranges] == [10, 13]
    assert [
        (item.start_index, item.end_index)
        for item in ranges
    ] == [
        (0, 3),
        (4, 7),
    ]

    chunks = StaticSpanChunker(
        [
            ChunkSpan(
                content="c\nx",
                start_index=2,
                end_index=5,
            )
        ]
    ).chunk_document(
        pages,
        ChunkingConfig(chunker_name="custom"),
    )

    assert chunks[0].page_ids == [10, 13]


def test_chunking_hash_depends_on_parsing_and_chunking_config() -> None:
    token_512 = ChunkingConfig(
        chunker_name="token",
        options={"chunk_size": 512},
    )
    token_256 = ChunkingConfig(
        chunker_name="token",
        options={"chunk_size": 256},
    )

    first = chunking_configuration_hash(token_512, "parse-a")

    assert first == chunking_configuration_hash(token_512, "parse-a")
    assert first != chunking_configuration_hash(token_256, "parse-a")
    assert first != chunking_configuration_hash(token_512, "parse-b")


def test_chunking_storage_round_trip(tmp_path) -> None:
    config = ChunkingConfig(
        chunker_name="token",
        options={"chunk_size": 512},
    )
    chunks = [
        Chunk(
            chunk_id="doc:00000001",
            doc_id="doc",
            page_ids=[11],
            content="second",
        ),
        Chunk(
            chunk_id="doc:00000000",
            doc_id="doc",
            page_ids=[10, 11],
            content="first",
            metadata={"backend": "test"},
        ),
    ]

    result = save_chunking_result(
        pdfs_path=tmp_path,
        config=config,
        parsing_config_hash="parse-a",
        chunks=chunks,
    )
    loaded = load_chunking_result(
        pdfs_path=tmp_path,
        config=config,
        parsing_config_hash="parse-a",
    )

    assert isinstance(result, ChunkingResult)
    assert result.output_path.exists()
    assert result.config_hash == loaded.config_hash
    assert [chunk.chunk_id for chunk in loaded.chunks] == [
        "doc:00000000",
        "doc:00000001",
    ]
    assert loaded.chunks[0].page_ids == [10, 11]
    assert loaded.chunks[0].metadata == {"backend": "test"}


def test_chonkie_adapter_normalizes_native_offsets(monkeypatch) -> None:
    @dataclass
    class NativeChunk:
        text: str
        start_index: int
        end_index: int
        token_count: int

    class FakeTokenChunker:
        def __init__(self, **options) -> None:
            self.options = options

        def chunk(self, text: str) -> list[NativeChunk]:
            assert text == "abc\ndef"
            return [
                NativeChunk(
                    text="bc\nd",
                    start_index=1,
                    end_index=5,
                    token_count=4,
                )
            ]

    monkeypatch.setattr(
        "ragit.chunking.chonkie._load_chonkie_chunker_class",
        lambda chunker_name: FakeTokenChunker,
    )

    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        ),
        _parsed_page(
            page_id=11,
            page_number=1,
            content="def",
            status=PageStatus.SUCCESS,
        ),
    ]

    chunks = ChonkieChunker().chunk_document(
        pages,
        ChunkingConfig(
            chunker_name="token",
            options={"chunk_size": 4},
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "bc\nd"
    assert chunks[0].page_ids == [10, 11]
    assert chunks[0].metadata["backend"] == "chonkie"
    assert chunks[0].metadata["chunker_name"] == "token"
    assert chunks[0].metadata["token_count"] == 4


def test_chonkie_accepts_non_serializable_runtime_options() -> None:
    callback = object()

    ChonkieChunker.validate_config(
        ChunkingConfig(
            chunker_name="token",
            options={"callback": callback},
        )
    )


def test_non_serializable_options_disable_cross_run_cache_reuse() -> None:
    first = ChunkingConfig(
        chunker_name="token",
        options={"callback": object()},
    )
    second = ChunkingConfig(
        chunker_name="token",
        options={"callback": object()},
    )

    with pytest.warns(RuntimeWarning, match="cache reuse is disabled"):
        first_hash = chunking_configuration_hash(first, "parse-a")
    with pytest.warns(RuntimeWarning, match="cache reuse is disabled"):
        second_hash = chunking_configuration_hash(second, "parse-a")

    with pytest.warns(RuntimeWarning, match="cache reuse is disabled"):
        repeated_first_hash = chunking_configuration_hash(first, "parse-a")

    assert first_hash == repeated_first_hash
    assert first_hash != second_hash


def test_cache_key_enables_stable_identity_for_runtime_objects() -> None:
    first = ChunkingConfig(
        chunker_name="token",
        options={"callback": object()},
        cache_key="callback-v1",
    )
    second = ChunkingConfig(
        chunker_name="token",
        options={"callback": object()},
        cache_key="callback-v1",
    )

    assert chunking_configuration_hash(
        first,
        "parse-a",
    ) == chunking_configuration_hash(second, "parse-a")


def test_known_chonkie_rules_receive_stable_cache_identity() -> None:
    @dataclass
    class RecursiveLevel:
        delimiters: list[str]
        whitespace: bool = False

    @dataclass
    class RecursiveRules:
        rules: list[RecursiveLevel]

    RecursiveLevel.__module__ = "chonkie.types"
    RecursiveRules.__module__ = "chonkie.types"

    first = ChunkingConfig(
        chunker_name="recursive",
        options={
            "rules": RecursiveRules([RecursiveLevel(["\n\n", ". "])])
        },
    )
    second = ChunkingConfig(
        chunker_name="recursive",
        options={
            "rules": RecursiveRules([RecursiveLevel(["\n\n", ". "])])
        },
    )

    assert chunking_configuration_hash(
        first,
        "parse-a",
    ) == chunking_configuration_hash(second, "parse-a")


def test_describe_chunker_inspects_runtime_options(monkeypatch) -> None:
    from ragit.chunking import describe_chunker, list_chunker_options

    class FakeSemanticChunker:
        def __init__(
            self,
            embedding_model: str = "fake/model",
            threshold: float = 0.8,
            **kwargs,
        ) -> None:
            pass

    monkeypatch.setattr(
        "ragit.chunking.chonkie._load_chonkie_chunker_class",
        lambda chunker_name: FakeSemanticChunker,
    )

    description = describe_chunker("semantic")

    assert description["options"]["embedding_model"]["default"] == "fake/model"
    assert description["options"]["threshold"]["default"] == 0.8
    assert description["accepts_additional_options"] is True
    assert description["documentation"].endswith("/semantic-chunker")
    assert list_chunker_options("semantic") == description["options"]


def test_invalid_chonkie_options_include_discovery_guidance(monkeypatch) -> None:
    from ragit.chunking import ChunkerConfigurationError, chunk_document

    class RejectingChunker:
        def __init__(self, **options) -> None:
            raise TypeError("unexpected keyword argument 'bad_option'")

    monkeypatch.setattr(
        "ragit.chunking.chonkie._load_chonkie_chunker_class",
        lambda chunker_name: RejectingChunker,
    )
    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        )
    ]

    with pytest.raises(
        ChunkerConfigurationError,
        match=r"describe_chunker\('token'\)",
    ):
        chunk_document(
            parsed_pages=pages,
            config=ChunkingConfig(
                chunker_name="token",
                options={"bad_option": True},
            ),
        )


@pytest.mark.parametrize(
    ("chunker_name", "options"),
    [
        ("token", {"chunk_size": 10}),
        ("sentence", {"chunk_size": 10}),
        ("recursive", {"chunk_size": 10}),
        ("fast", {"chunk_size": 10}),
        ("semantic", {"chunk_size": 10}),
        (
            "late",
            {"chunk_size": 10, "embedding_model": "fake/late-model"},
        ),
        ("neural", {"min_characters_per_chunk": 1}),
    ],
)
def test_public_api_resolves_all_registered_chonkie_chunkers(
    monkeypatch,
    chunker_name,
    options,
) -> None:
    from ragit.chunking import chunk_document

    @dataclass
    class NativeChunk:
        text: str
        start_index: int
        end_index: int
        token_count: int
        embedding: list[float] | None = None

    class FakeChonkieChunker:
        def __init__(self, **options) -> None:
            self.options = options

        def chunk(self, text: str) -> list[NativeChunk]:
            return [
                NativeChunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=len(text),
                    embedding=[1.0, 0.0] if chunker_name == "late" else None,
                )
            ]

    monkeypatch.setattr(
        "ragit.chunking.chonkie._load_chonkie_chunker_class",
        lambda selected_name: FakeChonkieChunker,
    )

    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        )
    ]

    chunks = chunk_document(
        parsed_pages=pages,
        config=ChunkingConfig(
            chunker_name=chunker_name,
            options=options,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "abc"
    assert chunks[0].page_ids == [10]
    assert chunks[0].metadata["chunker_name"] == chunker_name


def test_public_api_passes_options_to_selected_chonkie_chunker(monkeypatch) -> None:
    from ragit.chunking import chunk_document

    received_options = {}

    @dataclass
    class NativeChunk:
        text: str
        start_index: int
        end_index: int
        token_count: int

    class FakeTokenChunker:
        def __init__(self, **options) -> None:
            received_options.update(options)

        def chunk(self, text: str) -> list[NativeChunk]:
            return [
                NativeChunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=len(text),
                )
            ]

    monkeypatch.setattr(
        "ragit.chunking.chonkie._load_chonkie_chunker_class",
        lambda chunker_name: FakeTokenChunker,
    )

    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="abc",
            status=PageStatus.SUCCESS,
        )
    ]

    chunks = chunk_document(
        parsed_pages=pages,
        config=ChunkingConfig(
            chunker_name="token",
            options={"future_chonkie_option": True},
        ),
    )

    assert received_options == {"future_chonkie_option": True}
    assert len(chunks) == 1


def test_custom_registered_chunker_uses_same_public_api() -> None:
    from ragit.chunking import chunk_document, register_chunker

    class CustomChunker(BaseChunker):
        chunker_name = "test_custom_chunker"

        @classmethod
        def validate_config(cls, config: ChunkingConfig) -> None:
            super().validate_config(config)
            allowed = {"prefix_length"}
            unsupported = set(config.options) - allowed
            if unsupported:
                raise ValueError(
                    f"Unsupported options: {sorted(unsupported)}"
                )

        def _chunk_text(
            self,
            *,
            text: str,
            config: ChunkingConfig,
        ) -> list[ChunkSpan]:
            length = config.options.get("prefix_length", len(text))
            return [
                ChunkSpan(
                    content=text[:length],
                    start_index=0,
                    end_index=length,
                )
            ]

    register_chunker(CustomChunker)

    pages = [
        _parsed_page(
            page_id=42,
            page_number=0,
            content="abcdef",
            status=PageStatus.SUCCESS,
        )
    ]

    chunks = chunk_document(
        parsed_pages=pages,
        config=ChunkingConfig(
            chunker_name="test_custom_chunker",
            options={"prefix_length": 3},
        ),
    )

    assert chunks[0].content == "abc"
    assert chunks[0].page_ids == [42]


def test_function_chunker_maps_text_and_preserves_options() -> None:
    from ragit.chunking import chunk_document, chunker

    received_options = {}

    @chunker("test_paragraph_function")
    def split_paragraphs(text, options):
        received_options.update(options)
        return text.split("\n\n")

    chunks = chunk_document(
        parsed_pages=[
            _parsed_page(
                page_id=10,
                page_number=0,
                content="first paragraph\n\nsecond paragraph",
                status=PageStatus.SUCCESS,
            )
        ],
        config=ChunkingConfig(
            chunker_name="test_paragraph_function",
            options={"minimum_length": 5},
        ),
    )

    assert received_options == {"minimum_length": 5}
    assert [item.content for item in chunks] == [
        "first paragraph",
        "second paragraph",
    ]
    assert [item.page_ids for item in chunks] == [[10], [10]]


def test_function_chunker_accepts_explicit_spans_and_metadata() -> None:
    from ragit.chunking import ChunkSpan, chunk_document, chunker

    @chunker("test_explicit_span_function")
    def select_span(text, options):
        return [
            ChunkSpan(
                content=text[2:5],
                start_index=2,
                end_index=5,
                metadata={"section": options["section"]},
            )
        ]

    chunks = chunk_document(
        parsed_pages=[
            _parsed_page(
                page_id=10,
                page_number=0,
                content="abcdef",
                status=PageStatus.SUCCESS,
            )
        ],
        config=ChunkingConfig(
            chunker_name="test_explicit_span_function",
            options={"section": "middle"},
        ),
    )

    assert chunks[0].content == "cde"
    assert chunks[0].metadata == {"section": "middle"}


def test_public_api_rejects_unknown_chunker() -> None:
    from ragit.chunking import ChunkerRegistryError, chunk_document

    with pytest.raises(ChunkerRegistryError, match="Unknown chunker"):
        chunk_document(
            parsed_pages=[],
            config=ChunkingConfig(chunker_name="does_not_exist"),
        )


def test_page_chunker_creates_one_chunk_per_usable_page() -> None:
    from ragit.chunking import chunk_document

    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="first page",
            status=PageStatus.SUCCESS,
        ),
        _parsed_page(
            page_id=11,
            page_number=1,
            content="",
            status=PageStatus.EMPTY,
        ),
        _parsed_page(
            page_id=12,
            page_number=2,
            content=None,
            status=PageStatus.FAILED,
        ),
        _parsed_page(
            page_id=13,
            page_number=3,
            content="fourth page",
            status=PageStatus.SUCCESS,
        ),
    ]

    chunks = chunk_document(
        parsed_pages=pages,
        config=ChunkingConfig(
            chunker_name="page",
            options={},
        ),
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc:00000000",
        "doc:00000001",
    ]
    assert [chunk.page_ids for chunk in chunks] == [[10], [13]]
    assert [chunk.content for chunk in chunks] == [
        "first page",
        "fourth page",
    ]
    assert all(
        chunk.metadata == {
            "backend": "ragit",
            "chunker_name": "page",
        }
        for chunk in chunks
    )


def test_page_chunker_rejects_options() -> None:
    from ragit.chunking import chunk_document

    pages = [
        _parsed_page(
            page_id=10,
            page_number=0,
            content="page",
            status=PageStatus.SUCCESS,
        )
    ]

    with pytest.raises(ValueError, match="accepts no options"):
        chunk_document(
            parsed_pages=pages,
            config=ChunkingConfig(
                chunker_name="page",
                options={"chunk_size": 512},
            ),
        )
