"""Run dense indexing on real chunks from one PDF."""

from pathlib import Path

from ragit.chunking import (
    ChunkingConfig,
    chunk_document,
    save_chunking_result,
)
from ragit.data import initialize_collection
from ragit.indexing import IndexingConfig, build_index
from ragit.parsing import ParsingConfig, parse_document


DATA_PATH = Path("data/experiments_bench")
PDFS_PATH = DATA_PATH / "pdfs"

collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
)

document = collection.documents[0]

print(f"Document: {document.file_name}")
print(f"PDF path: {document.local_path}")

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

print("\nChunking document...")
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
print(f"Chunking hash: {chunking_result.config_hash}")

indexing_config = IndexingConfig(
    index_type="dense",
    model_name="BAAI/bge-m3",
    options={
        "batch_size": 16,
        "show_progress_bar": True,
    },
)

print("\nBuilding dense index...")
index_result = build_index(
    collection=collection,
    chunking_result=chunking_result,
    config=indexing_config,
)

print("\nDense indexing completed")
print(f"Output:           {index_result.output_path}")
print(f"Config hash:      {index_result.config_hash}")
print(f"Chunks indexed:   {len(index_result.chunk_mapping)}")
print(f"FAISS vectors:    {index_result.index.ntotal}")
print(f"Vector dimension: {index_result.index.d}")

print("\nFirst chunk mappings:")
for mapping in index_result.chunk_mapping[:10]:
    print(
        f"FAISS position {mapping.index_position}"
        f" -> chunk_id {mapping.chunk_id}"
    )
