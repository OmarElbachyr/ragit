"""Page-level evaluation for RAGit using ir-measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragit.data.models import Collection, Qrel
from ragit.evaluation.models import EvaluationResult, QueryEvaluation
from ragit.evaluation.storage import save_evaluation_result
from ragit.retrieval.models import RetrievalResult, RetrievedPage


SUPPORTED_METRICS = ("nDCG", "Recall", "Success", "RR")
_IR_METRIC_NAMES = {
    "nDCG": "nDCG",
    "Recall": "R",
    "Success": "Success",
    "RR": "RR",
}


class EvaluationConfigurationError(ValueError):
    """Raised when a page-level evaluation request is invalid."""


@dataclass(frozen=True)
class ResolvedMetric:
    """One user-facing metric/cutoff resolved to an ir-measures object."""

    family: str
    cutoff: int
    label: str
    measure: Any


def evaluate(
    *,
    collection: Collection,
    retrieval_result: RetrievalResult,
    metrics: list[str],
    k: list[int],
) -> EvaluationResult:
    """Evaluate ranked pages against canonical collection qrels."""
    metric_families = _validate_metrics(metrics)
    cutoffs = _validate_cutoffs(
        k,
        retrieval_top_k=getattr(retrieval_result, "top_k", None),
    )
    retrieval_query_ids = _validate_retrieval_result(
        collection=collection,
        retrieval_result=retrieval_result,
    )
    evaluated_qrels = _evaluated_qrels(
        collection=collection,
        retrieval_query_ids=retrieval_query_ids,
    )

    ir_measures = _load_ir_measures()
    resolved_metrics = resolve_metrics(
        metric_families,
        cutoffs,
        ir_measures=ir_measures,
    )
    ir_qrels = _to_ir_qrels(
        evaluated_qrels,
        ir_measures=ir_measures,
    )
    ir_run = _to_ir_run(
        retrieval_result.results,
        ir_measures=ir_measures,
    )

    measure_objects = [item.measure for item in resolved_metrics]
    aggregate_raw = ir_measures.calc_aggregate(
        measure_objects,
        ir_qrels,
        ir_run,
    )
    per_query_raw = list(
        ir_measures.iter_calc(
            measure_objects,
            ir_qrels,
            ir_run,
        )
    )

    aggregate = _normalize_aggregate(
        resolved_metrics=resolved_metrics,
        raw=aggregate_raw,
    )
    per_query = _normalize_per_query(
        resolved_metrics=resolved_metrics,
        raw=per_query_raw,
        evaluated_qrels=evaluated_qrels,
    )

    return save_evaluation_result(
        pdfs_path=collection.pdfs_path,
        retrieval_result=retrieval_result,
        qrels=evaluated_qrels,
        metrics=metric_families,
        k=cutoffs,
        aggregate=aggregate,
        per_query=per_query,
    )


def resolve_metrics(
    metrics: list[str],
    k: list[int],
    *,
    ir_measures: Any | None = None,
) -> list[ResolvedMetric]:
    """Resolve user metric families and cutoffs to ir-measures objects."""
    metric_families = _validate_metrics(metrics)
    cutoffs = _normalize_cutoffs(k)
    backend = ir_measures or _load_ir_measures()

    resolved: list[ResolvedMetric] = []
    for family in metric_families:
        ir_name = _IR_METRIC_NAMES[family]
        base_measure = getattr(backend, ir_name, None)
        if base_measure is None:
            raise EvaluationConfigurationError(
                f"ir-measures does not expose the required metric {ir_name!r}."
            )

        for cutoff in cutoffs:
            try:
                measure = base_measure @ cutoff
            except Exception as error:
                raise EvaluationConfigurationError(
                    f"ir-measures could not resolve {family}@{cutoff}."
                ) from error

            resolved.append(
                ResolvedMetric(
                    family=family,
                    cutoff=cutoff,
                    label=f"{family}@{cutoff}",
                    measure=measure,
                )
            )

    return resolved


def _validate_metrics(metrics: list[str]) -> list[str]:
    if not metrics:
        raise EvaluationConfigurationError(
            "At least one evaluation metric family is required."
        )

    unknown = sorted(set(metrics) - set(SUPPORTED_METRICS))
    if unknown:
        raise EvaluationConfigurationError(
            "Unsupported evaluation metric families: "
            f"{', '.join(unknown)}. Supported metrics: "
            f"{', '.join(SUPPORTED_METRICS)}."
        )

    requested = set(metrics)
    return [
        metric
        for metric in SUPPORTED_METRICS
        if metric in requested
    ]


def _normalize_cutoffs(k: list[int]) -> list[int]:
    if not k:
        raise EvaluationConfigurationError(
            "At least one evaluation cutoff k is required."
        )

    normalized: set[int] = set()
    for cutoff in k:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int):
            raise EvaluationConfigurationError(
                "Every evaluation cutoff k must be a positive integer."
            )
        if cutoff <= 0:
            raise EvaluationConfigurationError(
                "Every evaluation cutoff k must be a positive integer."
            )
        normalized.add(cutoff)

    return sorted(normalized)


def _validate_cutoffs(
    k: list[int],
    *,
    retrieval_top_k: int | None,
) -> list[int]:
    cutoffs = _normalize_cutoffs(k)
    if retrieval_top_k is None:
        raise EvaluationConfigurationError(
            "Retrieval result is missing its page-level top_k provenance."
        )
    if isinstance(retrieval_top_k, bool) or not isinstance(retrieval_top_k, int):
        raise EvaluationConfigurationError(
            "Retrieval result has an invalid page-level top_k value."
        )

    too_large = [cutoff for cutoff in cutoffs if cutoff > retrieval_top_k]
    if too_large:
        raise EvaluationConfigurationError(
            "Evaluation cutoff cannot exceed retrieval top_k. "
            f"retrieval top_k={retrieval_top_k}, invalid k={too_large}."
        )
    return cutoffs


def _validate_retrieval_result(
    *,
    collection: Collection,
    retrieval_result: RetrievalResult,
) -> set[int]:
    if not retrieval_result.results:
        raise EvaluationConfigurationError(
            "Evaluation requires a non-empty page-level retrieval result."
        )

    canonical_query_ids = {
        query.query_id for query in collection.queries
    }
    canonical_page_ids = {
        page.page_id for page in collection.pages
    }
    if not canonical_query_ids:
        raise EvaluationConfigurationError(
            "Evaluation requires collection queries."
        )
    if not canonical_page_ids:
        raise EvaluationConfigurationError(
            "Evaluation requires canonical collection pages."
        )

    query_ids: set[int] = set()
    seen_pages_by_query: dict[int, set[int]] = {}
    seen_ranks_by_query: dict[int, set[int]] = {}

    for result in retrieval_result.results:
        if not isinstance(result, RetrievedPage):
            raise EvaluationConfigurationError(
                "Evaluation requires page-level retrieval results."
            )
        if result.query_id not in canonical_query_ids:
            raise EvaluationConfigurationError(
                "Retrieval result references unknown canonical "
                f"query_id={result.query_id}."
            )
        if result.page_id not in canonical_page_ids:
            raise EvaluationConfigurationError(
                "Retrieval result references unknown canonical "
                f"page_id={result.page_id}."
            )

        page_ids = seen_pages_by_query.setdefault(result.query_id, set())
        if result.page_id in page_ids:
            raise EvaluationConfigurationError(
                "Retrieval result contains duplicate page_id="
                f"{result.page_id} for query_id={result.query_id}."
            )
        page_ids.add(result.page_id)

        ranks = seen_ranks_by_query.setdefault(result.query_id, set())
        if result.rank in ranks:
            raise EvaluationConfigurationError(
                "Retrieval result contains duplicate rank="
                f"{result.rank} for query_id={result.query_id}."
            )
        ranks.add(result.rank)
        query_ids.add(result.query_id)

    return query_ids


def _evaluated_qrels(
    *,
    collection: Collection,
    retrieval_query_ids: set[int],
) -> list[Qrel]:
    if not collection.qrels:
        raise EvaluationConfigurationError(
            "Evaluation requires non-empty collection qrels."
        )

    canonical_query_ids = {
        query.query_id for query in collection.queries
    }
    canonical_page_ids = {
        page.page_id for page in collection.pages
    }

    for qrel in collection.qrels:
        if qrel.query_id not in canonical_query_ids:
            raise EvaluationConfigurationError(
                f"Qrel references unknown query_id={qrel.query_id}."
            )
        if qrel.page_id not in canonical_page_ids:
            raise EvaluationConfigurationError(
                f"Qrel references unknown page_id={qrel.page_id}."
            )

    evaluated = sorted(
        (
            qrel
            for qrel in collection.qrels
            if qrel.query_id in retrieval_query_ids
        ),
        key=lambda qrel: (
            qrel.query_id,
            qrel.page_id,
            qrel.score,
        ),
    )

    if not evaluated:
        raise EvaluationConfigurationError(
            "No collection qrels correspond to the queries in the "
            "retrieval result."
        )
    return evaluated


def _to_ir_qrels(
    qrels: list[Qrel],
    *,
    ir_measures: Any,
) -> list[Any]:
    return [
        ir_measures.Qrel(
            str(qrel.query_id),
            str(qrel.page_id),
            qrel.score,
        )
        for qrel in qrels
    ]


def _to_ir_run(
    results: list[RetrievedPage],
    *,
    ir_measures: Any,
) -> list[Any]:
    ordered_results = sorted(
        results,
        key=lambda result: (
            result.query_id,
            result.rank,
            result.page_id,
        ),
    )
    return [
        ir_measures.ScoredDoc(
            str(result.query_id),
            str(result.page_id),
            result.score,
        )
        for result in ordered_results
    ]


def _normalize_aggregate(
    *,
    resolved_metrics: list[ResolvedMetric],
    raw: dict[Any, Any],
) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for item in resolved_metrics:
        if item.measure not in raw:
            raise EvaluationConfigurationError(
                f"ir-measures did not return aggregate metric {item.label}."
            )
        aggregate[item.label] = float(raw[item.measure])
    return aggregate


def _normalize_per_query(
    *,
    resolved_metrics: list[ResolvedMetric],
    raw: list[Any],
    evaluated_qrels: list[Qrel],
) -> list[QueryEvaluation]:
    measure_labels = {
        item.measure: item.label
        for item in resolved_metrics
    }
    label_order = [
        item.label for item in resolved_metrics
    ]
    canonical_by_string = {
        str(query_id): query_id
        for query_id in sorted(
            {qrel.query_id for qrel in evaluated_qrels}
        )
    }
    values: dict[int, dict[str, float]] = {
        query_id: {}
        for query_id in canonical_by_string.values()
    }

    for metric in raw:
        query_key = str(metric.query_id)
        if query_key not in canonical_by_string:
            raise EvaluationConfigurationError(
                "ir-measures returned an unknown query_id="
                f"{metric.query_id!r}."
            )
        if metric.measure not in measure_labels:
            raise EvaluationConfigurationError(
                "ir-measures returned an unexpected metric "
                f"{metric.measure!r}."
            )

        query_id = canonical_by_string[query_key]
        label = measure_labels[metric.measure]
        values[query_id][label] = float(metric.value)

    per_query: list[QueryEvaluation] = []
    for query_id in sorted(values):
        query_metrics = values[query_id]
        missing = [
            label
            for label in label_order
            if label not in query_metrics
        ]
        if missing:
            raise EvaluationConfigurationError(
                "ir-measures did not return all requested per-query metrics "
                f"for query_id={query_id}: missing {missing}."
            )

        per_query.append(
            QueryEvaluation(
                query_id=query_id,
                metrics={
                    label: query_metrics[label]
                    for label in label_order
                },
            )
        )

    return per_query


def _load_ir_measures() -> Any:
    try:
        import ir_measures
    except ImportError as error:
        raise ImportError(
            "ir-measures is required for RAGit evaluation."
        ) from error
    return ir_measures
