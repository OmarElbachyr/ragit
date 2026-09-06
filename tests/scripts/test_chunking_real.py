"""Run real document-level chunking on one parsed PDF."""

from collections import Counter
from pathlib import Path

import torch

from ragit.chunking import (
    ChunkingConfig,
    chunk_document,
    save_chunking_result,
)
from ragit.data import initialize_collection
from ragit.parsing import ParsingConfig, parse_document


DATA_PATH = Path("data/experiments_bench")
PDFS_PATH = DATA_PATH / "pdfs"

collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
)

document = collection.documents[0]

parsing_config = ParsingConfig(
    parser_name="pymupdf4llm",
    output_format="text",
    options={
        "use_ocr": True,
    },
)

print(f"Document: {document.file_name}")
print(f"PDF path: {document.local_path}")
print("Loading/parsing document...\n")

parsing_result = parse_document(
    collection=collection,
    doc_id=document.doc_id,
    config=parsing_config,
    overwrite=False,
)

parsed_pages = parsing_result.pages
statuses = Counter(page.status.value for page in parsed_pages)

print("Parsing result")
print(f"Pages:   {len(parsed_pages)}")
print(f"Success: {statuses['success']}")
print(f"Empty:   {statuses['empty']}")
print(f"Failed:  {statuses['failed']}")

if not torch.cuda.is_available():
    raise RuntimeError("This script requires a CUDA-capable GPU.")

chunking_config = ChunkingConfig(
    chunker_name="late",
    options={
        "embedding_model": "nomic-ai/modernbert-embed-base",
        "chunk_size": 512,
        "device": "cuda",
    },
)

print("\nStarting document-level chunking...\n")
print(f"GPU: {torch.cuda.get_device_name(0)}")

chunks = chunk_document(
    parsed_pages=parsed_pages,
    config=chunking_config,
)

chunking_result = save_chunking_result(
    pdfs_path=PDFS_PATH,
    config=chunking_config,
    parsing_config_hash=parsing_result.config_hash,
    chunks=chunks,
    replace=True,
)

single_page_chunks = [
    chunk for chunk in chunks if len(chunk.page_ids) == 1
]
multi_page_chunks = [
    chunk for chunk in chunks if len(chunk.page_ids) > 1
]

print("Chunking completed")
print(f"Output: {chunking_result.output_path}")
print(f"Config hash: {chunking_result.config_hash}")
print(f"Chunks: {len(chunks)}")
print(f"Single-page chunks: {len(single_page_chunks)}")
print(f"Multi-page chunks:  {len(multi_page_chunks)}")

if chunks:
    chunk = chunks[0]
    print("\nFirst chunk")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page IDs: {chunk.page_ids}")
    print(f"Metadata: {chunk.metadata}")
    print("\nContent:")
    print(chunk.content[:1000])

if multi_page_chunks:
    chunk = multi_page_chunks[0]
    print("\nFirst multi-page chunk")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page IDs: {chunk.page_ids}")
    print(f"Metadata: {chunk.metadata}")
    print("\nContent:")
    print(chunk.content[:1500])
else:
    print("\nNo multi-page chunks were produced.")
