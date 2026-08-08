"""Run dense page-level retrieval on one selected PDF."""

from pathlib import Path

from ragit.chunking import ChunkingConfig, chunk_document, save_chunking_result
from ragit.data import initialize_collection
from ragit.indexing import IndexingConfig, build_index
from ragit.parsing import ParsingConfig, parse_document
from ragit.retrieval import retrieve


DATA_PATH = Path("data/vidore_v3_finance_en")
PDFS_PATH = DATA_PATH / "pdfs"

NUM_QUERIES = 5
TOP_K = 10  # number of pages returned
AGGREGATION = "max"

collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
    queries_path=DATA_PATH / "queries.jsonl",
    qrels_path=DATA_PATH / "qrels.tsv",
)

document = collection.documents[0]

print(f"Document: {document.file_name}")
print(f"PDF path: {document.local_path}")

page_by_id = {page.page_id: page for page in collection.pages}
query_by_id = {query.query_id: query for query in collection.queries}

document_qrels = [
    qrel
    for qrel in collection.qrels
    if (
        qrel.page_id in page_by_id
        and page_by_id[qrel.page_id].doc_id == document.doc_id
    )
]

query_ids = list(dict.fromkeys(qrel.query_id for qrel in document_qrels))
document_queries = [
    query_by_id[query_id]
    for query_id in query_ids
    if query_id in query_by_id
]

print(f"Queries for PDF: {len(document_queries)}")
print(f"Qrels for PDF:   {len(document_qrels)}")

selected_queries = document_queries[:NUM_QUERIES]
if not selected_queries:
    raise RuntimeError(
        f"No queries with qrels found for {document.file_name!r}."
    )

print(f"Queries tested:  {len(selected_queries)}")

parsing_config = ParsingConfig(
    parser_name="pymupdf4llm",
    output_format="text",
    options={"use_ocr": True},
)

print("\nLoading/parsing document...")
parsing_result = parse_document(
    collection=collection,
    doc_id=document.doc_id,
    config=parsing_config,
    overwrite=False,
)

chunking_config = ChunkingConfig(
    chunker_name="token",
    options={
        "tokenizer": "character",
        "chunk_size": 1024,
    },
)

print("Chunking document...")
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

print(f"Chunks: {len(chunking_result.chunks)}")

indexing_config = IndexingConfig(
    index_type="dense",
    model_name="BAAI/bge-m3",
    options={
        "batch_size": 16,
        "show_progress_bar": True,
    },
)

print("\nLoading/building dense index...")
index_result = build_index(
    collection=collection,
    chunking_result=chunking_result,
    config=indexing_config,
    overwrite=False,
)

print(f"FAISS vectors: {index_result.index.ntotal}")

retrieval_collection = collection.model_copy(
    update={"queries": selected_queries}
)

print(
    f"\nRunning retrieval "
    f"(top_k={TOP_K} pages, aggregation={AGGREGATION})..."
)
retrieval_result = retrieve(
    collection=retrieval_collection,
    index_result=index_result,
    top_k=TOP_K,
    aggregation=AGGREGATION,
)

print(f"Retrieved pages: {len(retrieval_result.results)}")
print(f"Output: {retrieval_result.output_path}")

for query in selected_queries:
    relevant_page_ids = [
        qrel.page_id
        for qrel in document_qrels
        if qrel.query_id == query.query_id
    ]

    query_results = [
        result
        for result in retrieval_result.results
        if result.query_id == query.query_id
    ]

    print(f"\nQuery {query.query_id}: {query.text}")
    print(f"Relevant pages: {relevant_page_ids}")

    for page in query_results:
        hit = page.page_id in relevant_page_ids
        best_chunk = min(page.chunks, key=lambda chunk: chunk.rank)

        print(
            f"  {page.rank}. "
            f"page={page.page_id}  "
            f"score={page.score:.4f}  "
            f"chunks={len(page.chunks)}  "
            f"best_chunk={best_chunk.chunk_id} "
            f"(chunk_rank={best_chunk.rank})  "
            f"{'HIT' if hit else '-'}"
        )
