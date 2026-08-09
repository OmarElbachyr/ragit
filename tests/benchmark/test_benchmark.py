"""Focused tests for explicit multi-experiment benchmark orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragit.benchmark import (
    BenchmarkExecutionError,
    benchmark_configuration_hash,
    run_benchmark,
)
from ragit.chunking import ChunkingConfig
from ragit.evaluation import EvaluationConfig
from ragit.experiment import ExperimentConfig, experiment_configuration_hash
from ragit.indexing import IndexingConfig
from ragit.parsing import ParsingConfig
from ragit.retrieval import RetrievalConfig


def _config(*, chunk_size: int = 1024) -> ExperimentConfig:
    return ExperimentConfig(
        parsing=ParsingConfig(
            parser_name="pymupdf4llm",
            output_format="text",
            options={"use_ocr": True},
        ),
        chunking=ChunkingConfig(
            chunker_name="token",
            options={"tokenizer": "character", "chunk_size": chunk_size},
        ),
        indexing=IndexingConfig(
            index_type="dense",
            model_name="BAAI/bge-m3",
            options={"batch_size": 16},
        ),
        retrieval=RetrievalConfig(top_k=20, aggregation="max"),
        evaluation=EvaluationConfig(
            metrics=["nDCG", "Recall", "Success", "RR"],
            k=[1, 5, 10],
        ),
    )


def _metrics(offset: float) -> dict[str, float]:
    values = {}
    current = offset
    for metric in ("nDCG", "Recall", "Success", "RR"):
        for cutoff in (1, 5, 10):
            values[f"{metric}@{cutoff}"] = current
            current += 0.01
    return values


def _fake_result(config: ExperimentConfig, tmp_path: Path, offset: float):
    experiment_hash = experiment_configuration_hash(config)
    return SimpleNamespace(
        experiment_hash=experiment_hash,
        metrics=_metrics(offset),
        output_path=tmp_path / "experiments" / experiment_hash,
    )


def test_runs_multiple_experiments_sequentially_and_preserves_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ragit.benchmark.runner as runner

    configs = {
        "token-512": _config(chunk_size=512),
        "token-1024": _config(chunk_size=1024),
    }
    calls = []

    def fake_run_experiment(*, collection, config, overwrite_parsing=False):
        calls.append((config.chunking.options["chunk_size"], overwrite_parsing))
        return _fake_result(config, tmp_path, 0.1 * len(calls))

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)
    collection = SimpleNamespace(pdfs_path=tmp_path / "pdfs")

    result = run_benchmark(
        collection=collection,
        experiments=configs,
        overwrite_parsing=True,
    )

    assert calls == [(512, True), (1024, True)]
    assert [item.label for item in result.experiments] == [
        "token-512",
        "token-1024",
    ]
    assert result.experiments[0].metrics == _metrics(0.1)
    assert result.experiments[1].metrics == _metrics(0.2)


def test_overwrite_parsing_defaults_to_false(monkeypatch, tmp_path: Path) -> None:
    import ragit.benchmark.runner as runner

    config = _config()
    seen = []

    def fake_run_experiment(*, collection, config, overwrite_parsing=False):
        seen.append(overwrite_parsing)
        return _fake_result(config, tmp_path, 0.1)

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)
    run_benchmark(
        collection=SimpleNamespace(pdfs_path=tmp_path / "pdfs"),
        experiments={"baseline": config},
    )
    assert seen == [False]


def test_benchmark_hash_is_deterministic_and_label_sensitive() -> None:
    first = _config(chunk_size=512)
    second = _config(chunk_size=1024)
    experiments = {"a": first, "b": second}

    assert benchmark_configuration_hash(experiments) == benchmark_configuration_hash(
        {"a": first, "b": second}
    )
    assert benchmark_configuration_hash(experiments) != benchmark_configuration_hash(
        {"renamed": first, "b": second}
    )


def test_label_does_not_change_individual_experiment_hash() -> None:
    config = _config()
    before = experiment_configuration_hash(config)
    benchmark_configuration_hash({"display-a": config})
    after = experiment_configuration_hash(config)
    benchmark_configuration_hash({"display-b": config})
    assert before == after


def test_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        benchmark_configuration_hash({"": _config()})


def test_rejects_mismatched_metric_families(tmp_path: Path) -> None:
    first = _config()
    second = _config().model_copy(
        update={
            "evaluation": EvaluationConfig(
                metrics=["nDCG", "Recall"],
                k=[1, 5, 10],
            )
        }
    )

    with pytest.raises(ValueError, match="metric families"):
        run_benchmark(
            collection=SimpleNamespace(pdfs_path=tmp_path / "pdfs"),
            experiments={"a": first, "b": second},
        )


def test_rejects_mismatched_k(tmp_path: Path) -> None:
    first = _config()
    second = _config().model_copy(
        update={
            "evaluation": EvaluationConfig(
                metrics=["nDCG", "Recall", "Success", "RR"],
                k=[1, 10],
            )
        }
    )

    with pytest.raises(ValueError, match="same evaluation k"):
        run_benchmark(
            collection=SimpleNamespace(pdfs_path=tmp_path / "pdfs"),
            experiments={"a": first, "b": second},
        )


def test_persists_config_results_and_stable_table(monkeypatch, tmp_path: Path) -> None:
    import ragit.benchmark.runner as runner

    first = _config(chunk_size=512)
    second = _config(chunk_size=1024)
    experiments = {"token-512": first, "token-1024": second}
    counter = {"value": 0}

    def fake_run_experiment(*, collection, config, overwrite_parsing=False):
        counter["value"] += 1
        return _fake_result(config, tmp_path, counter["value"] / 10)

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)
    result = run_benchmark(
        collection=SimpleNamespace(pdfs_path=tmp_path / "pdfs"),
        experiments=experiments,
    )

    config_path = result.output_path / "config.json"
    results_path = result.output_path / "results.json"
    table_path = result.output_path / "results.txt"
    assert config_path.exists()
    assert results_path.exists()
    assert table_path.exists()

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_results = json.loads(results_path.read_text(encoding="utf-8"))
    assert [item["label"] for item in saved_config["experiments"]] == [
        "token-512",
        "token-1024",
    ]
    assert saved_config["evaluation"] == first.evaluation.model_dump(mode="json")
    assert [item["label"] for item in saved_results["experiments"]] == [
        "token-512",
        "token-1024",
    ]

    header = result.results_table.splitlines()[1]
    expected = [
        "nDCG@1",
        "nDCG@5",
        "nDCG@10",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Success@1",
        "Success@5",
        "Success@10",
        "RR@1",
        "RR@5",
        "RR@10",
    ]
    positions = [header.index(label) for label in expected]
    assert positions == sorted(positions)
    assert table_path.read_text(encoding="utf-8") == result.results_table + "\n"


def test_rerun_reexecutes_all_experiments(monkeypatch, tmp_path: Path) -> None:
    import ragit.benchmark.runner as runner

    configs = {"a": _config(chunk_size=512), "b": _config(chunk_size=1024)}
    calls = {"count": 0}

    def fake_run_experiment(*, collection, config, overwrite_parsing=False):
        calls["count"] += 1
        return _fake_result(config, tmp_path, calls["count"] / 100)

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)
    collection = SimpleNamespace(pdfs_path=tmp_path / "pdfs")

    first = run_benchmark(collection=collection, experiments=configs)
    second = run_benchmark(collection=collection, experiments=configs)

    assert first.benchmark_hash == second.benchmark_hash
    assert calls["count"] == 4
    saved = json.loads(
        (second.output_path / "results.json").read_text(encoding="utf-8")
    )
    assert saved["experiments"][0]["metrics"] == _metrics(0.03)


def test_failing_experiment_is_identified_and_preserves_cause(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ragit.benchmark.runner as runner

    expected = RuntimeError("indexing exploded")

    def fake_run_experiment(*, collection, config, overwrite_parsing=False):
        if config.chunking.options["chunk_size"] == 1024:
            raise expected
        return _fake_result(config, tmp_path, 0.1)

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)

    with pytest.raises(BenchmarkExecutionError, match="token-1024") as error:
        run_benchmark(
            collection=SimpleNamespace(pdfs_path=tmp_path / "pdfs"),
            experiments={
                "token-512": _config(chunk_size=512),
                "token-1024": _config(chunk_size=1024),
            },
        )

    assert error.value.__cause__ is expected
