# Benchmarking

`run_benchmark` compares an explicit mapping of labels to `ExperimentConfig`
objects. Experiments run sequentially and must share the same evaluation
configuration so their aggregate metrics are directly comparable.

```python
from ragit.benchmark import run_benchmark

result = run_benchmark(
    collection=collection,
    experiments={
        "page": page_config,
        "token-512": token_config,
    },
)

print(result.results_table)
```

Each experiment persists its normal stage artifacts. The benchmark adds a
summary result and formatted comparison table under `.ragit/benchmarks/`.
