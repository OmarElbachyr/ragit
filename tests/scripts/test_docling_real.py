"""Run Docling on one real document."""

from collections import Counter
from pathlib import Path

from ragit.data import initialize_collection
from ragit.parsing import ParsingConfig, parse_document


DATA_PATH = Path("data/vidore_v3_finance_en")
PDFS_PATH = DATA_PATH / "pdfs"

collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
)

document = collection.documents[0]

document_page_count = sum(
    page.doc_id == document.doc_id
    for page in collection.pages
)

config = ParsingConfig(
    parser_name="docling",
    output_format="markdown",
    options={
        "use_ocr": True,
    },
)

print(f"Document: {document.file_name}")
print(f"PDF path: {document.local_path}")
print(f"Pages:    {document_page_count}")
print("Starting parsing...\n")

result = parse_document(
    collection=collection,
    doc_id=document.doc_id,
    config=config,
    overwrite=False,
)

pages = result.pages
statuses = Counter(page.status.value for page in pages)

print("\nParsing completed")
print(f"Output:   {result.output_path}")
print(f"Pages:    {len(pages)}")
print(f"Success:  {statuses['success']}")
print(f"Empty:    {statuses['empty']}")
print(f"Failed:   {statuses['failed']}")

successful_page = next(
    (
        page
        for page in pages
        if page.status.value == "success"
    ),
    None,
)

if successful_page:
    print("\nContent preview:")
    print(successful_page.content[:1000])

failed_page = next(
    (
        page
        for page in pages
        if page.status.value == "failed"
    ),
    None,
)

if failed_page:
    print("\nFirst error:")
    print(
        f"Type:    "
        f"{failed_page.metadata.get('error_type')}"
    )
    print(
        f"Message: "
        f"{failed_page.metadata.get('error_message')}"
    )