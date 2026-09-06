# Getting started

RAGit requires Python 3.10 or newer. From a local checkout, install it with:

```bash
pip install -e .
```

Install the `docling` extra if you want to use the Docling parser:

```bash
pip install -e '.[docling]'
```

Or install the `marker` extra for the Marker parser:

```bash
pip install -e '.[marker]'
```

The Marker extra includes vLLM for local open-source picture descriptions.
See [Marker](marker.md) for server startup and model selection.

```python
ParsingConfig(
    parser_name="docling",
    output_format="markdown",
    options={
        "use_ocr": True,
        "describe_pictures": True,
    },
)
```

See [Parsing](parsing.md) for parser capabilities, option semantics, and
additional configurations.

## Initialize a collection

At minimum, place one or more PDFs in a directory:

```python
from ragit.data import initialize_collection

collection = initialize_collection("data/my_collection/pdfs")
```

RAGit creates canonical document and page records under
`data/my_collection/pdfs/.ragit/`. To evaluate retrieval, also supply query and
qrels files as described in [Dataset format](dataset-format.md).

## Run an experiment

```python
from ragit.chunking import ChunkingConfig
from ragit.evaluation import EvaluationConfig
from ragit.experiment import ExperimentConfig, run_experiment
from ragit.indexing import IndexingConfig
from ragit.parsing import ParsingConfig
from ragit.retrieval import RetrievalConfig

config = ExperimentConfig(
    parsing=ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
    ),
    chunking=ChunkingConfig(
        chunker_name="page",
    ),
    indexing=IndexingConfig(
        index_type="dense",
        model_name="lightonai/mDenseOn",
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        aggregation="max",
    ),
    evaluation=EvaluationConfig(
        metrics=["nDCG", "Recall", "Success", "RR"],
        k=[1, 5, 10],
    ),
)

result = run_experiment(collection=collection, config=config)
print(result.metrics)
```

An experiment requires queries and page-level qrels in the initialized
collection. Results and reusable stage artifacts are written beneath `.ragit/`.

## Indexing

See [Indexing](indexing.md) for dense FAISS and late-interaction FastPlaid
configuration.
