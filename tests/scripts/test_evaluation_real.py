"""Run the full dense page-retrieval + evaluation pipeline on all PDFs."""

from pathlib import Path

from ragit.chunking import ChunkingConfig, chunk_document, save_chunking_result
from ragit.data import initialize_collection
from ragit.evaluation import evaluate
from ragit.indexing import IndexingConfig, build_index
from ragit.parsing import ParsingConfig, parse_document
from ragit.retrieval import retrieve


DATA_PATH = Path("data/experiments_bench")
PDFS_PATH = DATA_PATH / "pdfs"

TOP_K = 20
METRICS = ["nDCG", "Recall", "Success", "RR"]
K = [1, 5, 10]


collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
    queries_path=DATA_PATH / "queries.jsonl",
    qrels_path=DATA_PATH / "qrels.tsv",
)

print(f"Documents: {len(collection.documents)}")
print(f"Queries:   {len(collection.queries)}")
print(f"Qrels:     {len(collection.qrels)}")

parsing_config = ParsingConfig(
    parser_name="pymupdf4llm",
    output_format="text",
    options={"use_ocr": True},
)

chunking_config = ChunkingConfig(
    chunker_name="token",
    options={
        "tokenizer": "character",
        "chunk_size": 1024,
    },
)

all_chunks = []
parsing_config_hash = None

for i, document in enumerate(collection.documents, start=1):
    print(f"\n[{i}/{len(collection.documents)}] {document.file_name}")

    parsing_result = parse_document(
        collection=collection,
        doc_id=document.doc_id,
        config=parsing_config,
        overwrite=False,
    )

    if parsing_config_hash is None:
        parsing_config_hash = parsing_result.config_hash
    elif parsing_result.config_hash != parsing_config_hash:
        raise RuntimeError(
            "Parsing returned inconsistent configuration hashes across documents."
        )

    chunks = chunk_document(
        parsed_pages=parsing_result.pages,
        config=chunking_config,
    )

    print(f"Chunks: {len(chunks)}")
    all_chunks.extend(chunks)

if parsing_config_hash is None:
    raise RuntimeError("No documents were processed.")

print(f"\nTotal chunks: {len(all_chunks)}")

chunking_result = save_chunking_result(
    pdfs_path=collection.pdfs_path,
    config=chunking_config,
    parsing_config_hash=parsing_config_hash,
    chunks=all_chunks,
    replace=True,
)

index_result = build_index(
    collection=collection,
    chunking_result=chunking_result,
    config=IndexingConfig(
        index_type="dense",
        model_name="BAAI/bge-m3",
        options={
            "batch_size": 16,
            "show_progress_bar": True,
        },
    ),
)

retrieval_result = retrieve(
    collection=collection,
    index_result=index_result,
    top_k=TOP_K,
    aggregation="max",
)

evaluation_result = evaluate(
    collection=collection,
    retrieval_result=retrieval_result,
    metrics=METRICS,
    k=K,
)

print("\nEvaluation")
print(evaluation_result.results_table)
print(f"\nOutput: {evaluation_result.output_path}")
