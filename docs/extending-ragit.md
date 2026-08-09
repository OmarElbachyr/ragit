# Extending RAGit

RAGit exposes registries for parser and chunker implementations.

Parsers implement `BaseParser` and can be registered with `register_parser`.
Chunkers implement `BaseChunker` and can be registered with `register_chunker`.
Each implementation is selected by the name stored in its corresponding
Pydantic configuration model.

Extensions should return RAGit's normalized models and preserve document and
page provenance. Keeping that contract intact allows the existing persistence,
indexing, retrieval, and evaluation stages to consume custom results.

The extension API is under active development and may change before the first
stable release.
