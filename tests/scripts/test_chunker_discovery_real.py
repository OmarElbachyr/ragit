"""Exercise chunker discovery and contextual configuration errors."""

from pprint import pprint

from ragit.chunking import (
    ChunkerConfigurationError,
    ChunkingConfig,
    chunk_document,
    describe_chunker,
    list_chunker_options,
)
from ragit.parsing import PageStatus, ParsedPage


CHUNKER_NAMES = (
    "page",
    "token",
    "sentence",
    "recursive",
    "fast",
    "semantic",
    "late",
    "neural",
)
SEPARATOR = "=" * 80


print(SEPARATOR)
print("Inspecting supported chunkers")
print(SEPARATOR)

for chunker_name in CHUNKER_NAMES:
    description = describe_chunker(chunker_name)
    options = list_chunker_options(chunker_name)

    assert description["name"] == chunker_name
    assert description["options"] == options
    assert isinstance(description["documentation"], str)
    assert description["documentation"]

    print(f"\n{chunker_name}: {description['class']}")
    print(f"Documentation: {description['documentation']}")
    print("Options:")
    pprint(options)
    print(SEPARATOR)


parsed_pages = [
    ParsedPage(
        page_id=0,
        doc_id="discovery-check",
        page_number=0,
        content="A short document used to validate configuration errors.",
        content_format="text",
        status=PageStatus.SUCCESS,
        metadata={},
    )
]

try:
    chunk_document(
        parsed_pages=parsed_pages,
        config=ChunkingConfig(
            chunker_name="fast",
            options={"not_a_chonkie_option": True},
        ),
    )
except ChunkerConfigurationError as error:
    message = str(error)
    assert "describe_chunker('fast')" in message
    assert (
        "https://docs.chonkie.ai/oss/chunkers/fast-chunker"
        in message
    )
    print("\nContextual error check passed")
    print(SEPARATOR)
    print(message)
    print(SEPARATOR)
else:
    raise AssertionError("Invalid FastChunker options were unexpectedly accepted.")

print("\nChunker discovery checks passed.")
