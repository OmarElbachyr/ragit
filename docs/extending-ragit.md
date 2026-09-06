# Extending RAGit

RAGit offers three levels for custom chunking. Start with the smallest
interface that provides the control your strategy needs. In every case, the
custom implementation must be imported before its name is used in a
`ChunkingConfig`; registration is local to the running Python process.

## 1. Return strings for the common case

Use `@chunker` when the strategy can return pieces of the source text in their
original order:

```python
from ragit.chunking import ChunkingConfig, chunker

@chunker("paragraph")
def split_paragraphs(text, options):
    minimum_length = options.get("minimum_length", 1)
    return [
        paragraph
        for paragraph in text.split("\n\n")
        if len(paragraph) >= minimum_length
    ]

config = ChunkingConfig(
    chunker_name="paragraph",
    options={"minimum_length": 50},
)
```

RAGit locates each returned string by searching forward from the end of the
previous chunk. It then creates chunk IDs, determines source page IDs, and
preserves pipeline provenance. Empty strings and text that cannot be mapped
back to the source are rejected.

This automatic mapping is intentionally conservative. If chunks overlap,
repeat out of order, or modify the source text, return explicit spans instead.

## 2. Return spans for precise boundaries and metadata

The same decorator can return `ChunkSpan` objects when exact offsets or custom
metadata are needed:

```python
from ragit.chunking import ChunkSpan, chunker

@chunker("section")
def split_sections(text, options):
    return [
        ChunkSpan(
            content=text[10:120],
            start_index=10,
            end_index=120,
            metadata={"section_type": "summary"},
        )
    ]
```

A plain mapping with `content`, `start_index`, `end_index`, and optional
`metadata` fields is also accepted. The content must exactly match the stated
slice of the source text. RAGit still owns chunk IDs and page provenance.

## 3. Subclass BaseChunker for complete control

Advanced integrations can implement `BaseChunker` directly and register the
class:

```python
from ragit.chunking import (
    BaseChunker,
    ChunkSpan,
    register_chunker,
)

class CustomChunker(BaseChunker):
    chunker_name = "custom"

    @classmethod
    def validate_config(cls, config):
        super().validate_config(config)
        # Add strategy-specific validation here.

    def _chunk_text(self, *, text, config):
        return [
            ChunkSpan(
                content=text,
                start_index=0,
                end_index=len(text),
            )
        ]

register_chunker(CustomChunker)
```

Subclassing is appropriate when an external backend needs initialization,
custom validation, or a reusable adapter class. `BaseChunker` still handles
document reconstruction, normalized chunk creation, and page provenance.

## Shared requirements

- Chunker names must be non-empty and unique within the process.
- Runtime options may contain arbitrary Python objects. Provide
  `ChunkingConfig.cache_key` when those objects should reuse artifacts across
  runs; otherwise RAGit safely disables cross-run cache reuse.
- Returned content must come from the reconstructed source document.
- Explicit spans use half-open character offsets: `start_index` is included
  and `end_index` is excluded.
- Import the module containing a custom chunker before running an experiment.

Parsers have a separate advanced extension API: implement `BaseParser` and
register it with `register_parser`. Parser extensions must return RAGit's
normalized page models and preserve document/page provenance.

The extension API remains under active development and may change before the
first stable release.
