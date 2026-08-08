"""Persistence for RAGit page-level retrieval runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ragit.data.models import Query
from ragit.indexing.models import IndexResult
from ragit.retrieval.models import RetrievalResult, RetrievedPage


CONFIG_FILE = "config.json"
RESULTS_FILE = "results.json"


class RetrievalStorageError(ValueError):
    """Raised for invalid retrieval persistence metadata."""


def query_set_hash(queries: Iterable[Query]) -> str:
    normalized = sorted(
        (
            {"query_id": query.query_id, "text": query.text}
            for query in queries
        ),
        key=lambda item: item["query_id"],
    )
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def aggregation_identifier(aggregation: str | Callable[..., Any]) -> str:
    if isinstance(aggregation, str):
        return aggregation
    module = getattr(aggregation, "__module__", None)
    qualname = getattr(aggregation, "__qualname__", None)
    if module and qualname:
        return f"custom:{module}.{qualname}"
    aggregation_type = type(aggregation)
    return (
        "custom:"
        f"{aggregation_type.__module__}.{aggregation_type.__qualname__}"
    )


def retrieval_configuration_hash(
    *,
    index_result: IndexResult,
    queries: Iterable[Query],
    top_k: int,
    aggregation: str | Callable[..., Any] = "max",
) -> str:
    payload = _configuration_payload(
        index_result=index_result,
        queries=queries,
        top_k=top_k,
        aggregation=aggregation,
    )
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def retrieval_output_path(
    *,
    pdfs_path: str | Path,
    index_result: IndexResult,
    queries: Iterable[Query],
    top_k: int,
    aggregation: str | Callable[..., Any] = "max",
) -> Path:
    config_hash = retrieval_configuration_hash(
        index_result=index_result,
        queries=queries,
        top_k=top_k,
        aggregation=aggregation,
    )
    index_type = _index_type(index_result)
    return (
        Path(pdfs_path)
        / ".ragit"
        / "retrieval"
        / index_type
        / config_hash
    )


def save_retrieval_result(
    *,
    pdfs_path: str | Path,
    index_result: IndexResult,
    queries: Iterable[Query],
    top_k: int,
    aggregation: str | Callable[..., Any],
    results: Iterable[RetrievedPage],
) -> RetrievalResult:
    """Persist a recomputed page retrieval run, overwriting same identity."""
    query_list = list(queries)
    normalized_results = list(results)
    output_path = retrieval_output_path(
        pdfs_path=pdfs_path,
        index_result=index_result,
        queries=query_list,
        top_k=top_k,
        aggregation=aggregation,
    )
    output_path.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        output_path / CONFIG_FILE,
        _configuration_payload(
            index_result=index_result,
            queries=query_list,
            top_k=top_k,
            aggregation=aggregation,
        ),
    )
    _write_json_atomic(
        output_path / RESULTS_FILE,
        _results_payload(
            queries=query_list,
            results=normalized_results,
        ),
    )

    return RetrievalResult(
        results=normalized_results,
        config_hash=retrieval_configuration_hash(
            index_result=index_result,
            queries=query_list,
            top_k=top_k,
            aggregation=aggregation,
        ),
        output_path=output_path,
        top_k=top_k,
    )


def _results_payload(
    *,
    queries: list[Query],
    results: list[RetrievedPage],
) -> dict[str, Any]:
    results_by_query: dict[int, list[RetrievedPage]] = {
        query.query_id: [] for query in queries
    }
    for result in results:
        if result.query_id not in results_by_query:
            raise RetrievalStorageError(
                "Retrieval result references unknown "
                f"query_id={result.query_id}."
            )
        results_by_query[result.query_id].append(result)

    query_records: list[dict[str, Any]] = []
    for query in queries:
        pages = sorted(
            results_by_query[query.query_id],
            key=lambda page: page.rank,
        )
        query_records.append(
            {
                "query_id": query.query_id,
                "pages": [
                    {
                        "page_id": page.page_id,
                        "score": page.score,
                        "rank": page.rank,
                        "chunks": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "score": chunk.score,
                                "rank": chunk.rank,
                                "page_ids": chunk.page_ids,
                            }
                            for chunk in page.chunks
                        ],
                    }
                    for page in pages
                ],
            }
        )
    return {"queries": query_records}


def _configuration_payload(
    *,
    index_result: IndexResult,
    queries: Iterable[Query],
    top_k: int,
    aggregation: str | Callable[..., Any],
) -> dict[str, Any]:
    if index_result.config is None:
        raise RetrievalStorageError(
            "Index result is missing its indexing configuration."
        )

    query_list = list(queries)
    return {
        "retrieval_type": index_result.config.index_type,
        "index_config_hash": index_result.config_hash,
        "model_name": index_result.config.model_name,
        "top_k": top_k,
        "top_k_unit": "pages",
        "aggregation": aggregation_identifier(aggregation),
        "query_set_hash": query_set_hash(query_list),
    }


def _index_type(index_result: IndexResult) -> str:
    if index_result.config is None:
        raise RetrievalStorageError(
            "Index result is missing its indexing configuration."
        )
    return index_result.config.index_type


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
