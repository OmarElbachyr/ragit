# Chunking

RAGit supports its native `page` strategy and the Chonkie `token`, `sentence`,
`recursive`, `fast`, `semantic`, `late`, and `neural` strategies. Chonkie owns
the strategy-specific constructor options and validation; RAGit passes options
through unchanged, including Python objects such as custom tokenizers,
embedding providers, and recursive rules.

## Discover options at runtime

Use `describe_chunker` to inspect the installed Chonkie version rather than a
duplicated RAGit option list:

```python
from ragit.chunking import describe_chunker, list_chunker_options

description = describe_chunker("semantic")
print(description["options"])
print(description["documentation"])

options = list_chunker_options("semantic")
```

The description includes constructor parameters, defaults, whether the
constructor accepts additional keyword arguments, and its documentation URL.
Options forwarded through `**kwargs` cannot always be enumerated by Python
introspection; consult the linked Chonkie page when
`accepts_additional_options` is true.

## Strategies and official references

| Strategy | Purpose | Option reference |
| --- | --- | --- |
| `page` | One chunk per usable PDF page | RAGit-native; accepts no options |
| `token` | Fixed-size token chunks | [Chonkie TokenChunker](https://docs.chonkie.ai/oss/chunkers/token-chunker) |
| `sentence` | Sentence-preserving chunks | [Chonkie SentenceChunker](https://docs.chonkie.ai/oss/chunkers/sentence-chunker) |
| `recursive` | Hierarchical splitting for structured text | [Chonkie RecursiveChunker](https://docs.chonkie.ai/oss/chunkers/recursive-chunker) |
| `fast` | High-throughput byte-based splitting | [Chonkie FastChunker](https://docs.chonkie.ai/oss/chunkers/fast-chunker) |
| `semantic` | Similarity-based grouping | [Chonkie SemanticChunker](https://docs.chonkie.ai/oss/chunkers/semantic-chunker) |
| `late` | Context-aware chunk embeddings | [Chonkie LateChunker](https://docs.chonkie.ai/oss/chunkers/late-chunker) |
| `neural` | Neural semantic-shift detection | [Chonkie NeuralChunker](https://docs.chonkie.ai/oss/chunkers/neural-chunker) |

## Example configurations

These Python configurations adapt initialization examples from Chonkie's
official documentation.

### Page

```python
ChunkingConfig(chunker_name="page")
```

### Token

```python
ChunkingConfig(
    chunker_name="token",
    options={"tokenizer": "gpt2", "chunk_size": 512, "chunk_overlap": 50},
)
```

### Sentence

```python
ChunkingConfig(
    chunker_name="sentence",
    options={
        "tokenizer": "character",
        "chunk_size": 2048,
        "chunk_overlap": 128,
        "min_sentences_per_chunk": 1,
    },
)
```

### Recursive

```python
ChunkingConfig(
    chunker_name="recursive",
    options={
        "tokenizer": "character",
        "chunk_size": 2048,
        "min_characters_per_chunk": 24,
    },
)
```

Chonkie also accepts `RecursiveRules` objects. RAGit converts those known rule
objects into a stable cache representation while passing the original objects
to Chonkie at runtime.

### Fast

```python
ChunkingConfig(
    chunker_name="fast",
    options={"chunk_size": 4096, "delimiters": "\n.?"},
)
```

### Semantic

```python
ChunkingConfig(
    chunker_name="semantic",
    options={
        "embedding_model": "minishlab/potion-base-32M",
        "threshold": 0.8,
        "chunk_size": 2048,
        "similarity_window": 3,
        "skip_window": 0,
    },
)
```

### Late

```python
ChunkingConfig(
    chunker_name="late",
    options={
        "embedding_model": "nomic-ai/modernbert-embed-base",
        "chunk_size": 2048,
        "min_characters_per_chunk": 24,
    },
)
```

When `embedding_model` is a string, dense indexing verifies that the same model
is used so the context-aware embeddings are preserved. Other Chonkie-supported
embedding objects are also accepted.

### Neural

```python
ChunkingConfig(
    chunker_name="neural",
    options={
        "model": "mirth/chonky_modernbert_base_1",
        "device_map": "cpu",
        "min_characters_per_chunk": 10,
    },
)
```

## Configuration errors

If Chonkie rejects an option, RAGit reports the selected strategy, preserves
the original error as its cause, and points to both `describe_chunker` and the
official option reference.

## Caching runtime objects

RAGit uses the following cache policy without restricting Chonkie's runtime
API:

1. JSON values are included directly in the cache identity.
2. Known `RecursiveRules` and `RecursiveLevel` objects are converted into a
   stable representation that includes their fields and Chonkie version.
3. For custom tokenizers, models, callbacks, or other objects, set a stable
   `cache_key` on `ChunkingConfig` when equivalent objects should reuse cached
   artifacts.
4. Without a serializer or `cache_key`, execution still proceeds, but RAGit
   uses a per-run identity and warns that cross-run cache reuse is disabled.

The user-provided key must change whenever the behavior or state of the runtime
objects changes.

```python
config = ChunkingConfig(
    chunker_name="token",
    options={"tokenizer": custom_tokenizer},
    cache_key="company-tokenizer-v2",
)
```

## Custom strategies

RAGit supports function-based custom strategies for minimal boilerplate,
explicit spans for precise offsets and metadata, and `BaseChunker` subclasses
for complete control. See [Extending RAGit](extending-ragit.md) for examples
and the mapping rules.
