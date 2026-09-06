# Indexing

RAGit supports two local retrieval configurations. Both retrieve chunks first,
aggregate chunk scores to pages, and return the requested top-k pages.

| `index_type` | Encoder | Index | Retrieval behavior |
| --- | --- | --- | --- |
| `dense` | `SentenceTransformer` | FAISS `IndexFlatIP` | One normalized vector per chunk. |
| `late_interaction` | `MultiVectorEncoder` | FastPlaid | Token-level vectors with MaxSim scoring. |

## How page ranks are produced

RAGit always retrieves chunks before it ranks pages. A chunk can contribute to
each page listed in its metadata; RAGit applies the configured aggregation
function to those contributions and sorts the resulting page scores.

## Dense FAISS

Dense indexing encodes every chunk to one L2-normalized vector and stores it in
a FAISS `IndexFlatIP` index. Inner-product search over normalized vectors uses
cosine similarity.

```python
IndexingConfig(
    index_type="dense",
    model_name="lightonai/DenseOn",
    options={
        "device": "cuda",
        "batch_size": 64,
    },
)
```

Both index types expose the same model-encoding options:

| Option | Default | Purpose |
| --- | --- | --- |
| `batch_size` | `32` | Inputs encoded per model batch. |
| `device` | automatic | Model device, for example `"cuda"` or `"cpu"`. |
| `show_progress_bar` | `False` | Show model-encoding progress. |
| `trust_remote_code` | `False` | Allow a model repository's custom code. |

## Late interaction FastPlaid

Late-interaction models preserve one vector per token. RAGit uses a Sentence
Transformers `MultiVectorEncoder` to encode chunks and queries, then indexes
the document token matrices with FastPlaid for MaxSim retrieval.

```python
IndexingConfig(
    index_type="late_interaction",
    model_name="lightonai/LateOn",
    options={
        "device": "cuda",
        "batch_size": 32,
    },
)
```

FastPlaid uses its backend defaults for index construction and search. The
public model options are the same as for dense indexing.
