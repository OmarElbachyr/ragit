# RAGit

RAGit is a local Python library for comparing document preprocessing and
retrieval configurations with page-level evaluation.

Its end-to-end workflow covers collection initialization, PDF parsing,
chunking, local indexing, chunk retrieval, page aggregation, evaluation,
experiments, and benchmarks. Stage artifacts are stored under the collection's
`.ragit/` directory and reused when their configuration hashes match.

Start with [Getting started](getting-started.md), or review the
[dataset format](dataset-format.md) before importing an existing benchmark.

The API is under active development and is not yet stable.
