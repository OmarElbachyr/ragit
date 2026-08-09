"""Run one complete dense experiment on the initialized RAGit collection."""

from pathlib import Path

from ragit.chunking import ChunkingConfig
from ragit.data import initialize_collection
from ragit.evaluation import EvaluationConfig
from ragit.experiment import ExperimentConfig, run_experiment
from ragit.indexing import IndexingConfig
from ragit.parsing import ParsingConfig
from ragit.retrieval import RetrievalConfig


DATA_PATH = Path("data/experiments_bench")
PDFS_PATH = DATA_PATH / "pdfs"


collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
    queries_path=DATA_PATH / "queries.jsonl",
    qrels_path=DATA_PATH / "qrels.tsv",
)

config = ExperimentConfig(
    parsing=ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
        options={"use_ocr": True},
    ),
    chunking=ChunkingConfig(
        chunker_name="token",
        options={
            "tokenizer": "character",
            "chunk_size": 1024,
        },
    ),
    indexing=IndexingConfig(
        index_type="dense",
        model_name="BAAI/bge-m3",
        options={
            "batch_size": 16,
            "show_progress_bar": True,
        },
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

result = run_experiment(
    collection=collection,
    config=config,
    overwrite_parsing=False,
)

print("\nEvaluation")
for metric, value in result.metrics.items():
    print(f"{metric}: {value:.4f}")
print(f"\nOutput: {result.output_path}")
