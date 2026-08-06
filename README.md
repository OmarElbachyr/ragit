# ragit

**ragit** ("RAG it") is a local, open-source Python library for comparing document parsing and chunking configurations through page-level retrieval evaluation.

The project is at an early development stage. Its planned workflow is:

```text
PDFs
→ parsing
→ chunking
→ local indexing
→ chunk retrieval
→ page aggregation
→ page-level evaluation
```

Planned integrations include Chonkie for chunking, FAISS for dense indexes, PyLate for late-interaction retrieval, and `ir-measures` for evaluation.

The API is not stable, and the package is under active development.
