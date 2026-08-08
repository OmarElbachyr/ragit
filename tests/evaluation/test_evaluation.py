from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ragit.data.models import Collection, Page, Qrel, Query
from ragit.evaluation import (
    EvaluationConfigurationError,
    evaluate,
    resolve_metrics,
)
from ragit.retrieval.models import RetrievalResult, RetrievedChunk, RetrievedPage


@dataclass(frozen=True)
class FakeMeasure:
    name: str
    cutoff: int

    def __str__(self) -> str:
        return f"{self.name}@{self.cutoff}"


@dataclass(frozen=True)
class FakeFamily:
    name: str

    def __matmul__(self, cutoff: int) -> FakeMeasure:
        return FakeMeasure(self.name, cutoff)


@dataclass
class FakeQrel:
    query_id: str
    doc_id: str
    relevance: int


@dataclass
class FakeScoredDoc:
    query_id: str
    doc_id: str
    score: float


@dataclass
class FakeMetric:
    query_id: str
    measure: FakeMeasure
    value: float


class FakeIRMeasures:
    nDCG = FakeFamily("nDCG")
    R = FakeFamily("R")
    Success = FakeFamily("Success")
    RR = FakeFamily("RR")
    Qrel = FakeQrel
    ScoredDoc = FakeScoredDoc

    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.per_query_calls = 0
        self.last_qrels: list[FakeQrel] = []
        self.last_run: list[FakeScoredDoc] = []

    def calc_aggregate(self, measures, qrels, run):
        self.aggregate_calls += 1
        self.last_qrels = list(qrels)
        self.last_run = list(run)
        return {
            measure: (index + 1) / 10.0
            for index, measure in enumerate(measures)
        }

    def iter_calc(self, measures, qrels, run):
        self.per_query_calls += 1
        query_ids = sorted({qrel.query_id for qrel in qrels}, key=int)
        for query_id in query_ids:
            for index, measure in enumerate(measures):
                yield FakeMetric(
                    query_id=query_id,
                    measure=measure,
                    value=(int(query_id) + 1) + ((index + 1) / 100.0),
                )


def _collection(tmp_path: Path) -> Collection:
    return Collection(
        pdfs_path=tmp_path / "pdfs",
        ragit_path=tmp_path / "pdfs" / ".ragit",
        documents=[],
        pages=[
            Page(page_id=10, doc_id="doc", page_number=0),
            Page(page_id=11, doc_id="doc", page_number=1),
            Page(page_id=20, doc_id="doc", page_number=2),
            Page(page_id=21, doc_id="doc", page_number=3),
        ],
        queries=[
            Query(query_id=0, text="first query"),
            Query(query_id=1, text="second query"),
        ],
        qrels=[
            Qrel(query_id=0, page_id=10, score=2),
            Qrel(query_id=0, page_id=11, score=1),
            Qrel(query_id=1, page_id=20, score=1),
        ],
    )


def _retrieval_result(tmp_path: Path, top_k: int = 10) -> RetrievalResult:
    def page(query_id: int, page_id: int, score: float, rank: int) -> RetrievedPage:
        chunk = RetrievedChunk(
            query_id=query_id,
            chunk_id=f"doc:{query_id}:{rank}",
            page_ids=[page_id],
            score=score,
            rank=rank,
        )
        return RetrievedPage(
            query_id=query_id,
            page_id=page_id,
            score=score,
            rank=rank,
            chunks=[chunk],
        )

    return RetrievalResult(
        results=[
            page(0, 10, 0.9, 1),
            page(0, 11, 0.8, 2),
            page(1, 21, 0.9, 1),
            page(1, 20, 0.8, 2),
        ],
        config_hash="retrieval-hash",
        output_path=tmp_path / "retrieval",
        top_k=top_k,
    )


def test_resolve_metrics_has_deterministic_family_and_cutoff_order():
    fake = FakeIRMeasures()

    resolved = resolve_metrics(
        ["RR", "Recall", "nDCG"],
        [10, 1, 5],
        ir_measures=fake,
    )

    assert [item.label for item in resolved] == [
        "nDCG@1",
        "nDCG@5",
        "nDCG@10",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "RR@1",
        "RR@5",
        "RR@10",
    ]
    assert str(resolved[3].measure) == "R@1"


def test_unsupported_metric_is_rejected():
    with pytest.raises(EvaluationConfigurationError, match="Unsupported"):
        resolve_metrics(
            ["MAP"],
            [10],
            ir_measures=FakeIRMeasures(),
        )


@pytest.mark.parametrize("cutoffs", [[0], [-1], [True], [1.5], []])
def test_invalid_cutoffs_are_rejected(tmp_path, cutoffs):
    with pytest.raises(EvaluationConfigurationError):
        evaluate(
            collection=_collection(tmp_path),
            retrieval_result=_retrieval_result(tmp_path),
            metrics=["nDCG"],
            k=cutoffs,
        )


def test_cutoff_cannot_exceed_retrieval_top_k(tmp_path):
    with pytest.raises(EvaluationConfigurationError, match="cannot exceed"):
        evaluate(
            collection=_collection(tmp_path),
            retrieval_result=_retrieval_result(tmp_path, top_k=5),
            metrics=["nDCG"],
            k=[10],
        )


def test_qrels_and_run_conversion_preserve_canonical_ids_and_grades(
    tmp_path,
    monkeypatch,
):
    fake = FakeIRMeasures()
    import ragit.evaluation.evaluator as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "_load_ir_measures",
        lambda: fake,
    )

    evaluate(
        collection=_collection(tmp_path),
        retrieval_result=_retrieval_result(tmp_path),
        metrics=["nDCG"],
        k=[1],
    )

    assert [(q.query_id, q.doc_id, q.relevance) for q in fake.last_qrels] == [
        ("0", "10", 2),
        ("0", "11", 1),
        ("1", "20", 1),
    ]
    assert [(r.query_id, r.doc_id, r.score) for r in fake.last_run] == [
        ("0", "10", 0.9),
        ("0", "11", 0.8),
        ("1", "21", 0.9),
        ("1", "20", 0.8),
    ]


def test_aggregate_per_query_persistence_table_and_overwrite(
    tmp_path,
    monkeypatch,
):
    fake = FakeIRMeasures()
    import ragit.evaluation.evaluator as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "_load_ir_measures",
        lambda: fake,
    )

    first = evaluate(
        collection=_collection(tmp_path),
        retrieval_result=_retrieval_result(tmp_path),
        metrics=["RR", "nDCG", "Recall", "Success"],
        k=[5, 1],
    )

    assert list(first.aggregate) == [
        "nDCG@1",
        "nDCG@5",
        "Recall@1",
        "Recall@5",
        "Success@1",
        "Success@5",
        "RR@1",
        "RR@5",
    ]
    assert [record.query_id for record in first.per_query] == [0, 1]
    assert list(first.per_query[0].metrics) == list(first.aggregate)

    config_path = first.output_path / "config.json"
    results_path = first.output_path / "results.json"
    table_path = first.output_path / "results.txt"

    assert config_path.exists()
    assert results_path.exists()
    assert table_path.exists()

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert list(saved["aggregate"]) == list(first.aggregate)
    assert [record["query_id"] for record in saved["per_query"]] == [0, 1]

    table = table_path.read_text(encoding="utf-8")
    assert "nDCG" in table
    assert "Recall" in table
    assert "Success" in table
    assert "RR" in table
    assert "|   1 |" in table
    assert "|   5 |" in table

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["retrieval_hash"] == "retrieval-hash"
    assert config["retrieval_top_k"] == 10
    assert config["metrics"] == ["nDCG", "Recall", "Success", "RR"]
    assert config["k"] == [1, 5]
    assert config["qrels_identity"]

    results_path.write_text("stale\n", encoding="utf-8")
    table_path.write_text("stale\n", encoding="utf-8")

    second = evaluate(
        collection=_collection(tmp_path),
        retrieval_result=_retrieval_result(tmp_path),
        metrics=["nDCG", "Recall", "Success", "RR"],
        k=[1, 5],
    )

    assert second.output_path == first.output_path
    assert fake.aggregate_calls == 2
    assert fake.per_query_calls == 2
    assert results_path.read_text(encoding="utf-8") != "stale\n"
    assert table_path.read_text(encoding="utf-8") != "stale\n"


def test_actual_ir_measures_simple_page_level_example(tmp_path):
    pytest.importorskip("ir_measures")

    result = evaluate(
        collection=_collection(tmp_path),
        retrieval_result=_retrieval_result(tmp_path),
        metrics=["nDCG", "Recall", "Success", "RR"],
        k=[1],
    )

    assert result.aggregate["nDCG@1"] == pytest.approx(0.5)
    assert result.aggregate["Recall@1"] == pytest.approx(0.25)
    assert result.aggregate["Success@1"] == pytest.approx(0.5)
    assert result.aggregate["RR@1"] == pytest.approx(0.5)
