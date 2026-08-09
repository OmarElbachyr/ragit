# Dataset format

RAGit works with a PDF directory and up to four canonical data files. Explicit
input files are copied into `<pdfs_path>/.ragit/` during initialization.

## `corpus.jsonl`

One JSON object per PDF page:

```json
{"page_id": 0, "doc_id": "report", "page_number": 0}
```

`page_id` must be globally unique. `page_number` is zero-based and must refer to
an existing page in the document.

## `documents_metadata.jsonl`

One JSON object per PDF:

```json
{"file_name": "report.pdf", "doc_id": "report", "local_path": "data/pdfs/report.pdf", "page_number": 12}
```

Additional metadata fields are preserved. If corpus or document metadata is
omitted, RAGit generates the missing file from the PDFs.

## `queries.jsonl`

One JSON object per query:

```json
{"query_id": 1, "text": "What was total revenue?"}
```

Additional query fields are preserved.

## `qrels.tsv`

Tab-separated page-level relevance judgments with a header:

```text
query_id\tpage_id\tscore
1\t7\t1
```

Queries and qrels are optional, but they must be supplied together. They are
required for experiment evaluation and benchmarking.
