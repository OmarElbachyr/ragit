"""Focused tests for single-experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragit.chunking import ChunkingConfig
from ragit.evaluation import EvaluationConfig
from ragit.experiment import (
    ExperimentConfig,
    experiment_configuration_hash,
    run_experiment,
)
from ragit.indexing import IndexingConfig
from ragit.parsing import ParsingConfig
from ragit.retrieval import RetrievalConfig


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        parsing=ParsingConfig(
            parser_name="pymupdf4llm",
            output_format="text",
            options={"use_ocr": True},
        ),
        chunking=ChunkingConfig(
            chunker_name="token",
            options={"tokenizer": "character", "chunk_size": 1024},
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


def _replace(config: ExperimentConfig, stage: str, **updates) -> ExperimentConfig:
    stage_config = getattr(config, stage).model_copy(update=updates)
    return config.model_copy(update={stage: stage_config})


def test_experiment_config_groups_stage_configs() -> None:
    config = _config()
    assert config.parsing.parser_name == "pymupdf4llm"
    assert config.chunking.chunker_name == "token"
    assert config.indexing.index_type == "dense"
    assert config.retrieval.top_k == 20
    assert config.evaluation.k == [1, 5, 10]


def test_experiment_hash_is_deterministic() -> None:
    config = _config()
    assert experiment_configuration_hash(config) == experiment_configuration_hash(
        ExperimentConfig.model_validate(config.model_dump())
    )


def test_experiment_hash_supports_runtime_chunking_objects() -> None:
    first = _config()
    first.chunking.options["callback"] = object()
    first.chunking.cache_key = "callback-v1"

    second = _config()
    second.chunking.options["callback"] = object()
    second.chunking.cache_key = "callback-v1"

    assert experiment_configuration_hash(first) == experiment_configuration_hash(
        second
    )


@pytest.mark.parametrize(
    ("stage", "updates"),
    [
        ("parsing", {"output_format": "markdown"}),
        ("chunking", {"options": {"tokenizer": "character", "chunk_size": 512}}),
        ("indexing", {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}),
        ("retrieval", {"top_k": 50}),
        ("retrieval", {"aggregation": "mean"}),
        ("evaluation", {"metrics": ["nDCG"]}),
        ("evaluation", {"k": [1, 10]}),
    ],
)
def test_each_stage_config_changes_experiment_hash(stage, updates) -> None:
    config = _config()
    changed = _replace(config, stage, **updates)
    assert experiment_configuration_hash(config) != experiment_configuration_hash(
        changed
    )


def test_run_experiment_orchestrates_existing_stages_and_persists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ragit.experiment.runner as runner

    config = _config()
    collection = SimpleNamespace(
        pdfs_path=tmp_path / "pdfs",
        documents=[
            SimpleNamespace(doc_id="doc-a"),
            SimpleNamespace(doc_id="doc-b"),
        ],
    )
    calls = []

    def fake_parse_document(*, collection, doc_id, config, overwrite):
        calls.append(("parse", doc_id, overwrite))
        return SimpleNamespace(
            pages=[f"page:{doc_id}"],
            config_hash="parse-hash",
            output_path=tmp_path / "parse",
        )

    def fake_chunk_document(*, parsed_pages, config):
        calls.append(("chunk", tuple(parsed_pages)))
        return [f"chunk:{parsed_pages[0]}"]

    def fake_save_chunking_result(**kwargs):
        calls.append(("save_chunking", tuple(kwargs["chunks"]), kwargs["replace"]))
        assert kwargs["parsing_config_hash"] == "parse-hash"
        return SimpleNamespace(
            config_hash="chunk-hash",
            output_path=tmp_path / "chunk",
            chunks=list(kwargs["chunks"]),
        )

    def fake_build_index(**kwargs):
        assert "overwrite" not in kwargs
        calls.append(("index",))
        return SimpleNamespace(
            config_hash="index-hash",
            output_path=tmp_path / "index",
        )

    def fake_retrieve(**kwargs):
        calls.append(
            ("retrieve", kwargs["top_k"], kwargs["aggregation"])
        )
        return SimpleNamespace(
            config_hash="retrieval-hash",
            output_path=tmp_path / "retrieval",
        )

    def fake_evaluate(**kwargs):
        calls.append(
            ("evaluate", tuple(kwargs["metrics"]), tuple(kwargs["k"]))
        )
        return SimpleNamespace(
            config_hash="evaluation-hash",
            output_path=tmp_path / "evaluation",
            aggregate={"nDCG@10": 0.72, "Recall@10": 0.81},
        )

    monkeypatch.setattr(runner, "parse_document", fake_parse_document)
    monkeypatch.setattr(runner, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(runner, "save_chunking_result", fake_save_chunking_result)
    monkeypatch.setattr(runner, "build_index", fake_build_index)
    monkeypatch.setattr(runner, "retrieve", fake_retrieve)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)

    result = run_experiment(
        collection=collection,
        config=config,
        overwrite_parsing=True,
    )

    assert calls == [
        ("parse", "doc-a", True),
        ("parse", "doc-b", True),
        ("chunk", ("page:doc-a",)),
        ("chunk", ("page:doc-b",)),
        (
            "save_chunking",
            ("chunk:page:doc-a", "chunk:page:doc-b"),
            True,
        ),
        ("index",),
        ("retrieve", 20, "max"),
        ("evaluate", ("nDCG", "Recall", "Success", "RR"), (1, 5, 10)),
    ]
    assert result.metrics == {"nDCG@10": 0.72, "Recall@10": 0.81}
    assert result.parsing.config_hash == "parse-hash"
    assert result.chunking.config_hash == "chunk-hash"
    assert result.indexing.config_hash == "index-hash"
    assert result.retrieval.config_hash == "retrieval-hash"
    assert result.evaluation.config_hash == "evaluation-hash"

    config_path = result.output_path / "config.json"
    result_path = result.output_path / "result.json"
    assert config_path.exists()
    assert result_path.exists()

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_config == config.model_dump(mode="json")
    assert saved_result["experiment_hash"] == result.experiment_hash
    assert saved_result["metrics"] == result.metrics
    assert set(saved_result["artifacts"]) == {
        "parsing",
        "chunking",
        "indexing",
        "retrieval",
        "evaluation",
    }
    assert "config" not in saved_result


def test_rerun_overwrites_manifest_and_recomputes_downstream(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import ragit.experiment.runner as runner

    config = _config()
    collection = SimpleNamespace(
        pdfs_path=tmp_path / "pdfs",
        documents=[SimpleNamespace(doc_id="doc-a")],
    )
    counts = {name: 0 for name in ("parse", "chunk", "save", "index", "retrieve", "evaluate")}

    def fake_parse_document(**kwargs):
        assert kwargs["overwrite"] is False
        counts["parse"] += 1
        return SimpleNamespace(
            pages=["page"],
            config_hash="parse-hash",
            output_path=tmp_path / "parse",
        )

    def fake_chunk_document(**kwargs):
        counts["chunk"] += 1
        return ["chunk"]

    def fake_save_chunking_result(**kwargs):
        counts["save"] += 1
        return SimpleNamespace(
            chunks=["chunk"],
            config_hash="chunk-hash",
            output_path=tmp_path / "chunk",
        )

    def fake_build_index(**kwargs):
        counts["index"] += 1
        return SimpleNamespace(
            config_hash="index-hash",
            output_path=tmp_path / "index",
        )

    def fake_retrieve(**kwargs):
        counts["retrieve"] += 1
        return SimpleNamespace(
            config_hash="retrieval-hash",
            output_path=tmp_path / "retrieval",
        )

    def fake_evaluate(**kwargs):
        counts["evaluate"] += 1
        return SimpleNamespace(
            config_hash="evaluation-hash",
            output_path=tmp_path / "evaluation",
            aggregate={"nDCG@10": counts["evaluate"] / 10},
        )

    monkeypatch.setattr(runner, "parse_document", fake_parse_document)
    monkeypatch.setattr(runner, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(runner, "save_chunking_result", fake_save_chunking_result)
    monkeypatch.setattr(runner, "build_index", fake_build_index)
    monkeypatch.setattr(runner, "retrieve", fake_retrieve)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)

    first = run_experiment(collection=collection, config=config)
    second = run_experiment(collection=collection, config=config)

    assert first.experiment_hash == second.experiment_hash
    assert counts == {
        "parse": 2,
        "chunk": 2,
        "save": 2,
        "index": 2,
        "retrieve": 2,
        "evaluate": 2,
    }
    saved = json.loads(
        (second.output_path / "result.json").read_text(encoding="utf-8")
    )
    assert saved["metrics"]["nDCG@10"] == 0.2


def test_run_experiment_does_not_accept_overwrite(tmp_path: Path) -> None:
    collection = SimpleNamespace(
        pdfs_path=tmp_path / "pdfs",
        documents=[SimpleNamespace(doc_id="doc-a")],
    )

    with pytest.raises(TypeError, match="overwrite"):
        run_experiment(
            collection=collection,
            config=_config(),
            overwrite=True,
        )


def test_stage_errors_propagate(monkeypatch, tmp_path: Path) -> None:
    import ragit.experiment.runner as runner

    expected = RuntimeError("parser exploded")

    def fail_parse(**kwargs):
        raise expected

    monkeypatch.setattr(runner, "parse_document", fail_parse)
    collection = SimpleNamespace(
        pdfs_path=tmp_path / "pdfs",
        documents=[SimpleNamespace(doc_id="doc-a")],
    )

    with pytest.raises(RuntimeError) as error:
        run_experiment(collection=collection, config=_config())

    assert error.value is expected
