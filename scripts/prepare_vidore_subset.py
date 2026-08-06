"""Prepare a ViDoRe-v3 subset for ragit experiments.

Required packages:
    pip install datasets huggingface_hub pypdf
"""

import csv
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from pypdf import PdfReader


DATASET_ID = "vidore/vidore_v3_finance_en"
OUTPUT_DIR = Path("data") / DATASET_ID.rsplit("/", 1)[-1]
OVERWRITE = True


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Refuse accidental overwrite
# ---------------------------------------------------------------------

if OUTPUT_DIR.exists():
    if not OVERWRITE:
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Set OVERWRITE = True to replace it."
        )

    shutil.rmtree(OUTPUT_DIR)

pdf_dir = OUTPUT_DIR / "pdfs"
pdf_dir.mkdir(parents=True)


# ---------------------------------------------------------------------
# Load the configured ViDoRe-v3 subset
# ---------------------------------------------------------------------

print(f"Loading {DATASET_ID}...")

corpus_dataset = load_dataset(
    DATASET_ID,
    "corpus",
    split="test",
)

queries_dataset = load_dataset(
    DATASET_ID,
    "queries",
    split="test",
)

qrels_dataset = load_dataset(
    DATASET_ID,
    "qrels",
    split="test",
)

documents_dataset = load_dataset(
    DATASET_ID,
    "documents_metadata",
    split="test",
)


# ---------------------------------------------------------------------
# Download original PDFs
# ---------------------------------------------------------------------

print("Downloading original PDFs...")

snapshot_dir = Path(
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=["pdfs/*.pdf"],
    )
)

downloaded_pdf_dir = snapshot_dir / "pdfs"

if not downloaded_pdf_dir.exists():
    raise FileNotFoundError("The dataset PDF directory was not downloaded.")

for source_pdf in sorted(downloaded_pdf_dir.glob("*.pdf")):
    shutil.copy2(source_pdf, pdf_dir / source_pdf.name)


# ---------------------------------------------------------------------
# Map ViDoRe document IDs to local PDFs
# ---------------------------------------------------------------------

doc_to_pdf = {}

for document in documents_dataset:
    doc_id = str(document["doc_id"])
    file_name = Path(document["file_name"]).name
    pdf_path = pdf_dir / file_name

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF for document {doc_id!r} does not exist: {pdf_path}"
        )

    if doc_id in doc_to_pdf:
        raise ValueError(f"Duplicate document ID: {doc_id}")

    doc_to_pdf[doc_id] = pdf_path


# ---------------------------------------------------------------------
# Read PDF page counts
# ---------------------------------------------------------------------

pdf_page_counts = {
    doc_id: len(PdfReader(pdf_path).pages)
    for doc_id, pdf_path in doc_to_pdf.items()
}


# ---------------------------------------------------------------------
# Create documents_metadata.jsonl
# ---------------------------------------------------------------------

documents_metadata_records = []

for document in documents_dataset:
    doc_id = str(document["doc_id"])
    pdf_path = doc_to_pdf[doc_id]

    documents_metadata_records.append(
        {
            "file_name": pdf_path.name,
            "doc_id": doc_id,
            "local_path": str(pdf_path),
            "doc_type": document.get("doc_type"),
            "doc_language": document.get("doc_language"),
            "visual_types": document.get("visual_types"),
            "page_number": pdf_page_counts[doc_id],
        }
    )

documents_metadata_records.sort(
    key=lambda record: record["doc_id"]
)


# ---------------------------------------------------------------------
# Create corpus.jsonl
# ---------------------------------------------------------------------

corpus_records = []
page_ids = set()

for page in corpus_dataset:
    page_id = int(page["corpus_id"])
    doc_id = str(page["doc_id"])
    page_number = int(page["page_number_in_doc"])

    if page_id in page_ids:
        raise ValueError(f"Duplicate page ID: {page_id}")

    if doc_id not in doc_to_pdf:
        raise ValueError(f"Unknown document ID in corpus: {doc_id}")

    if not 0 <= page_number < pdf_page_counts[doc_id]:
        raise ValueError(
            f"Invalid page number {page_number} for {doc_id}. "
            f"PDF contains {pdf_page_counts[doc_id]} pages."
        )

    page_ids.add(page_id)

    corpus_records.append(
        {
            "page_id": page_id,
            "doc_id": doc_id,
            "page_number": page_number,
        }
    )

corpus_records.sort(key=lambda record: record["page_id"])


# ---------------------------------------------------------------------
# Create queries.jsonl
# Keep only English queries.
# ---------------------------------------------------------------------

query_records = []
query_ids = set()

for query in queries_dataset:
    query_id = int(query["query_id"])

    if query_id in query_ids:
        raise ValueError(f"Duplicate query ID: {query_id}")

    query_ids.add(query_id)

    record = {
        "query_id": query_id,
        "text": query["query"].strip(),
        "language": query["language"],
    }

    answer = query.get("answer")

    if answer:
        record["answer"] = answer

    query_records.append(record)

query_records.sort(key=lambda record: record["query_id"])


# ---------------------------------------------------------------------
# Create qrels.tsv
# Preserve ViDoRe relevance grades.
# ---------------------------------------------------------------------

qrel_records = []
qrel_pairs = set()

for qrel in qrels_dataset:
    query_id = int(qrel["query_id"])

    # Ignore qrels belonging to non-English query translations.
    if query_id not in query_ids:
        continue

    page_id = int(qrel["corpus_id"])
    score = int(qrel["score"])

    if page_id not in page_ids:
        raise ValueError(f"Qrel references unknown page: {page_id}")

    pair = (query_id, page_id)

    if pair in qrel_pairs:
        raise ValueError(f"Duplicate qrel pair: {pair}")

    qrel_pairs.add(pair)
    qrel_records.append((query_id, page_id, score))

qrel_records.sort(key=lambda row: (row[0], row[1]))


# ---------------------------------------------------------------------
# Final consistency checks
# ---------------------------------------------------------------------

for query_id, page_id, _ in qrel_records:
    if query_id not in query_ids:
        raise ValueError(f"Qrel references unknown query: {query_id}")

    if page_id not in page_ids:
        raise ValueError(f"Qrel references unknown page: {page_id}")


# ---------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------

write_jsonl(
    OUTPUT_DIR / "documents_metadata.jsonl",
    documents_metadata_records,
)

write_jsonl(
    OUTPUT_DIR / "corpus.jsonl",
    corpus_records,
)

write_jsonl(
    OUTPUT_DIR / "queries.jsonl",
    query_records,
)

with (OUTPUT_DIR / "qrels.tsv").open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["query_id", "page_id", "score"])
    writer.writerows(qrel_records)


print(f"Created:   {OUTPUT_DIR}")
print(f"Documents: {len(documents_metadata_records)}")
print(f"Pages:     {len(corpus_records)}")
print(f"Queries:   {len(query_records)}")
print(f"Qrels:     {len(qrel_records)}")
