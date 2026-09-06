"""Run the full late-interaction page-retrieval + evaluation pipeline on one PDF."""

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

document = collection.documents[0]
print(f"Document: {document.file_name}")

page_by_id = {page.page_id: page for page in collection.pages}
query_by_id = {query.query_id: query for query in collection.queries}

document_qrels = [
    qrel
    for qrel in collection.qrels
    if qrel.page_id in page_by_id
    and page_by_id[qrel.page_id].doc_id == document.doc_id
]
query_ids = list(dict.fromkeys(qrel.query_id for qrel in document_qrels))
document_queries = [
    query_by_id[query_id]
    for query_id in query_ids
    if query_id in query_by_id
]
if not document_queries:
    raise RuntimeError(
        f"No queries with qrels found for {document.file_name!r}."
    )

print(f"Queries: {len(document_queries)}")
print(f"Qrels:   {len(document_qrels)}")

parsing_result = parse_document(
    collection=collection,
    doc_id=document.doc_id,
    config=ParsingConfig(
        parser_name="pymupdf4llm",
        output_format="text",
        options={"use_ocr": True},
    ),
    overwrite=False,
)

chunking_config = ChunkingConfig(
    chunker_name="token",
    options={
        "tokenizer": "character",
        "chunk_size": 1024,
    },
)
chunks = chunk_document(
    parsed_pages=parsing_result.pages,
    config=chunking_config,
)
chunking_result = save_chunking_result(
    pdfs_path=collection.pdfs_path,
    config=chunking_config,
    parsing_config_hash=parsing_result.config_hash,
    chunks=chunks,
    replace=True,
)

index_result = build_index(
    collection=collection,
    chunking_result=chunking_result,
    config=IndexingConfig(
        index_type="late_interaction",
        model_name="lightonai/GTE-ModernColBERT-v1",
        options={
            "batch_size": 16,
            "show_progress_bar": True,
        },
    ),
    overwrite=False,
)

subset = collection.model_copy(
    update={
        "queries": document_queries,
        "qrels": document_qrels,
    }
)

retrieval_result = retrieve(
    collection=subset,
    index_result=index_result,
    top_k=TOP_K,
    aggregation="mean",
)

evaluation_result = evaluate(
    collection=subset,
    retrieval_result=retrieval_result,
    metrics=METRICS,
    k=K,
)

print("\nEvaluation")
print(evaluation_result.results_table)
print(f"\nOutput: {evaluation_result.output_path}")