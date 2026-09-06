# RAGit

**RAGit** ("RAG it") is a local, open-source Python library for comparing
document parsing, chunking, indexing, and retrieval configurations through
page-level retrieval evaluation.

RAGit provides an end-to-end experimental workflow:

```text
PDFs
→ parsing
→ chunking
→ local indexing
→ chunk retrieval
→ page aggregation
→ page-level evaluation
```

Current capabilities include:

- canonical local collections with optional queries and page-level relevance judgments;
- Docling, Marker, and PyMuPDF4LLM parser adapters, with OCR and Docling
  picture descriptions;
- native page chunking plus Chonkie token, sentence, recursive, fast,
  semantic, late, and neural strategies;
- Sentence Transformers dense and multi-vector encoders, with FAISS dense
  indexes and FastPlaid late-interaction indexes;
- chunk-to-page score aggregation and evaluation with `ir-measures`;
- persisted, content-addressed artifacts under each collection's `.ragit/`
  directory; and
- single-experiment and multi-experiment benchmark orchestration.

## Installation

RAGit requires Python 3.10 or newer. Install the project for local development:

```bash
pip install -e .
```

Install a parser-specific extra when needed, for example:

```bash
pip install -e '.[docling]'
```

For Marker parsing:

```bash
pip install -e '.[marker]'
```

## Getting started

Initialize a collection from a directory of PDFs:

```python
from ragit.data import initialize_collection

collection = initialize_collection(pdfs_path="data/my_collection/pdfs")
```

For an evaluated collection, pass `corpus.jsonl`, `documents_metadata.jsonl`,
`queries.jsonl`, and `qrels.tsv` when initializing it. See the
[getting-started guide](docs/getting-started.md) for a complete experiment and
[dataset format](docs/dataset-format.md) for the expected records.
See [Chunking](docs/chunking.md) for supported strategies, runtime option
discovery, official Chonkie references, and example configurations.

## Project status

The API is not stable, and the package is under active development.
