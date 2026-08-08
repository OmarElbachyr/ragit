"""Deterministic persistence for RAGit page-level evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tabulate import tabulate

from ragit.data.models import Qrel
from ragit.evaluation.models import EvaluationResult, QueryEvaluation
from ragit.retrieval.models import RetrievalResult


CONFIG_FILE = "config.json"
RESULTS_FILE = "results.json"
RESULTS_TABLE_FILE = "results.txt"


class EvaluationStorageError(ValueError):
    """Raised for invalid evaluation persistence metadata."""


def qrels_identity(qrels: Iterable[Qrel]) -> str:
    """Return a deterministic identity for the evaluated qrels."""
    normalized = sorted(
        (
            {
                "query_id": qrel.query_id,
                "page_id": qrel.page_id,
                "score": qrel.score,
            }
            for qrel in qrels
        ),
        key=lambda item: (
            item["query_id"],
            item["page_id"],
            item["score"],
        ),
    )
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def evaluation_configuration_hash(
    *,
    retrieval_result: RetrievalResult,
    qrels: Iterable[Qrel],
    metrics: Iterable[str],
    k: Iterable[int],
) -> str:
    """Hash retrieval provenance, qrels, metric families, and cutoffs."""
    payload = _configuration_payload(
        retrieval_result=retrieval_result,
        qrels=qrels,
        metrics=metrics,
        k=k,
    )
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def evaluation_output_path(
    *,
    pdfs_path: str | Path,
    retrieval_result: RetrievalResult,
    qrels: Iterable[Qrel],
    metrics: Iterable[str],
    k: Iterable[int],
) -> Path:
    config_hash = evaluation_configuration_hash(
        retrieval_result=retrieval_result,
        qrels=qrels,
        metrics=metrics,
        k=k,
    )
    return Path(pdfs_path) / ".ragit" / "evaluation" / config_hash


def save_evaluation_result(
    *,
    pdfs_path: str | Path,
    retrieval_result: RetrievalResult,
    qrels: Iterable[Qrel],
    metrics: list[str],
    k: list[int],
    aggregate: dict[str, float],
    per_query: list[QueryEvaluation],
) -> EvaluationResult:
    """Persist a newly computed evaluation run, overwriting same identity."""
    qrel_list = list(qrels)
    output_path = evaluation_output_path(
        pdfs_path=pdfs_path,
        retrieval_result=retrieval_result,
        qrels=qrel_list,
        metrics=metrics,
        k=k,
    )
    output_path.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        output_path / CONFIG_FILE,
        _configuration_payload(
            retrieval_result=retrieval_result,
            qrels=qrel_list,
            metrics=metrics,
            k=k,
        ),
    )
    _write_json_atomic(
        output_path / RESULTS_FILE,
        {
            "aggregate": aggregate,
            "per_query": [
                record.model_dump(mode="json")
                for record in per_query
            ],
        },
    )

    results_table = format_results_table(
        aggregate=aggregate,
        metrics=metrics,
        k=k,
    )
    _write_text_atomic(
        output_path / RESULTS_TABLE_FILE,
        results_table + "\n",
    )

    return EvaluationResult(
        aggregate=aggregate,
        per_query=per_query,
        config_hash=evaluation_configuration_hash(
            retrieval_result=retrieval_result,
            qrels=qrel_list,
            metrics=metrics,
            k=k,
        ),
        output_path=output_path,
        results_table=results_table,
    )


def format_results_table(
    *,
    aggregate: dict[str, float],
    metrics: list[str],
    k: list[int],
) -> str:
    """Format aggregate metrics with cutoffs as rows and families as columns."""
    headers = ["k", *metrics]
    rows: list[list[Any]] = []

    for cutoff in k:
        row: list[Any] = [cutoff]
        for metric in metrics:
            label = f"{metric}@{cutoff}"
            if label not in aggregate:
                raise EvaluationStorageError(
                    f"Aggregate result is missing metric {label!r}."
                )
            row.append(aggregate[label])
        rows.append(row)

    return tabulate(
        rows,
        headers=headers,
        tablefmt="grid",
        floatfmt=".4f",
    )


def _configuration_payload(
    *,
    retrieval_result: RetrievalResult,
    qrels: Iterable[Qrel],
    metrics: Iterable[str],
    k: Iterable[int],
) -> dict[str, Any]:
    retrieval_top_k = getattr(retrieval_result, "top_k", None)
    if retrieval_top_k is None:
        raise EvaluationStorageError(
            "Retrieval result is missing its page-level top_k provenance."
        )

    metric_list = list(metrics)
    cutoff_list = list(k)
    qrel_list = list(qrels)
    return {
        "retrieval_hash": retrieval_result.config_hash,
        "retrieval_top_k": retrieval_top_k,
        "metrics": metric_list,
        "k": cutoff_list,
        "qrels_identity": qrels_identity(qrel_list),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    temporary_path.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
