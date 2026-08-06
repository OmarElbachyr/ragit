"""Run Docling on one real document."""

from collections import Counter
from pathlib import Path

from ragit.data import initialize_collection
from ragit.parsing import ParsingConfig, save_parsing_result
from ragit.parsing.parsers import DoclingParser


DATA_PATH = Path("data/vidore_v3_finance_en")
PDFS_PATH = DATA_PATH / "pdfs"

collection = initialize_collection(
    pdfs_path=PDFS_PATH,
    corpus_path=DATA_PATH / "corpus.jsonl",
    documents_metadata_path=DATA_PATH / "documents_metadata.jsonl",
)

document = collection.documents[0]

source_pages = sorted(
    [
        page
        for page in collection.pages
        if page.doc_id == document.doc_id
    ],
    key=lambda page: page.page_number,
)

config = ParsingConfig(
    parser_name="docling",
    output_format="markdown",
    options={
        "use_ocr": True,
    },
)

pages = DoclingParser().parse_document(
    document=document,
    source_pages=source_pages,
    config=config,
)

result = save_parsing_result(
    pdfs_path=PDFS_PATH,
    config=config,
    pages=pages,
    replace=True,
)

statuses = Counter(page.status.value for page in pages)

print(f"Document: {document.file_name}")
print(f"Output:   {result.output_path}")
print(f"Pages:    {len(pages)}")
print(f"Success:  {statuses['success']}")
print(f"Empty:    {statuses['empty']}")
print(f"Failed:   {statuses['failed']}")

successful_page = next(
    (page for page in pages if page.status.value == "success"),
    None,
)

if successful_page:
    print("\nContent preview:")
    print(successful_page.content[:1000])