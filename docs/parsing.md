# Parsing

RAGit parses PDF collections into one normalized `ParsedPage` record for each
source page. The parser output is either plain `text` or `markdown`; it is
then available to chunking and retrieval. A page is recorded as `success`,
`empty`, or `failed`, so a failure for one page does not stop the rest of the
document or collection.

## Built-in parsers

| Parser | Install | Output formats | Purpose |
| --- | --- | --- | --- |
| `pymupdf4llm` | Included | `text`, `markdown` | Lightweight local PDF text, Markdown, table, and OCR extraction |
| `docling` | `pip install -e '.[docling]'` | `text`, `markdown` | Structured PDF conversion with OCR, tables, whole-document VLM parsing, and optional picture descriptions |
| `marker` | `pip install -e '.[marker]'` | `markdown` | High-quality document conversion with layout analysis, OCR, tables, math, and optional LLM enhancement |

All parsers extract the PDF text layer and use their normal backend table
parsing configuration by default. Table parsing is not a public RAGit option.

## Configuration

`ParsingConfig` has three fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `parser_name` | Yes | The registered parser to run, such as `pymupdf4llm`, `docling`, or `marker`. |
| `output_format` | Yes | The normalized output format: `text` or `markdown`. |
| `options` | No | Parser-specific settings. Unsupported settings are rejected. |

`output_format` is independent of the parsing pipeline except for Marker,
which only produces Markdown. It otherwise always controls
the returned `ParsedPage.content` format: OCR can return either text or
Markdown, and Docling's VLM pipeline can also return either text or Markdown.
For example, `use_ocr=True` does **not** imply `output_format="text"`, and
`use_vlm=True` does **not** imply `output_format="markdown"`.

## Parser options and capabilities

Options are passed through `ParsingConfig.options`. Each adapter validates its
own fixed option set and rejects unsupported options.

| Parser | `use_ocr` | `use_vlm` | `describe_pictures` |
| --- | --- | --- | --- |
| `pymupdf4llm` | Yes | No | No |
| `docling` | Yes | Yes | Yes* |
| `marker` | Yes | No | Yes |

`use_ocr` is disabled by default. Enable it to recover literal text from
scanned or image-based regions of a page. Choose `output_format` separately
to receive that OCR-derived content as `text` or `markdown`.

`use_vlm` is available only in Docling. It selects Docling's whole-document
vision-language-model conversion pipeline rather than its standard PDF/OCR
pipeline. Choose `output_format` separately to receive its content as `text`
or `markdown`.

`describe_pictures` is available in Docling and Marker and is disabled by
default. It adds semantic descriptions of detected figures, diagrams, photos,
and other pictures to the exported content. It does not create image
embeddings. Marker uses a local open-source vision model for this approximate
mapping; see [Marker](marker.md) for the one-time server setup and model
selection.

`*` `describe_pictures` and `use_vlm` cannot be enabled together. They use
different Docling processing paths. `describe_pictures` can be combined with
`use_ocr`: OCR captures literal visual text, while picture descriptions add
semantic visual context.

## Example configurations

Basic local Markdown extraction:

```python
from ragit.parsing import ParsingConfig

config = ParsingConfig(
    parser_name="pymupdf4llm",
    output_format="markdown",
)
```

Docling with OCR and picture descriptions:

```python
config = ParsingConfig(
    parser_name="docling",
    output_format="markdown",
    options={
        "use_ocr": True,
        "describe_pictures": True,
    },
)
```

Docling whole-document VLM parsing:

```python
config = ParsingConfig(
    parser_name="docling",
    output_format="markdown",
    options={"use_vlm": True},
)
```

Marker Markdown extraction:

```python
config = ParsingConfig(
    parser_name="marker",
    output_format="markdown",
)
```

Marker with picture descriptions:

```python
config = ParsingConfig(
    parser_name="marker",
    output_format="markdown",
    options={"describe_pictures": True},
)
```

## Custom parsers

For a new backend, implement `BaseParser` and register it with
`register_parser`. Custom parsers must return RAGit's normalized page models
and preserve document and page provenance. See [Extending RAGit](extending-ragit.md).
