"""Deterministic identity and lightweight persistence for experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ragit.experiment.models import ExperimentConfig, ExperimentResult


CONFIG_FILE = "config.json"
RESULT_FILE = "result.json"


def experiment_configuration_hash(config: ExperimentConfig) -> str:
    """Return a deterministic hash of the complete normalized experiment config."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig instance.")

    payload = config.model_dump(mode="json")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def experiment_output_path(
    pdfs_path: str | Path,
    config: ExperimentConfig,
) -> Path:
    """Return the deterministic output directory for one experiment config."""
    return (
        Path(pdfs_path)
        / ".ragit"
        / "experiments"
        / experiment_configuration_hash(config)
    )


def save_experiment_result(
    *,
    pdfs_path: str | Path,
    result: ExperimentResult,
) -> ExperimentResult:
    """Persist the latest lightweight manifest for a completed experiment."""
    expected_hash = experiment_configuration_hash(result.config)
    if result.experiment_hash != expected_hash:
        raise ValueError(
            "ExperimentResult experiment_hash does not match its configuration."
        )

    output_path = experiment_output_path(pdfs_path, result.config)
    if result.output_path != output_path:
        result = result.model_copy(update={"output_path": output_path})

    output_path.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        output_path / CONFIG_FILE,
        result.config.model_dump(mode="json"),
    )
    _write_json_atomic(
        output_path / RESULT_FILE,
        _result_payload(result),
    )
    return result


def _result_payload(result: ExperimentResult) -> dict[str, Any]:
    """Return the lightweight persisted result payload."""
    return {
        "experiment_hash": result.experiment_hash,
        "artifacts": {
            "parsing": result.parsing.model_dump(mode="json"),
            "chunking": result.chunking.model_dump(mode="json"),
            "indexing": result.indexing.model_dump(mode="json"),
            "retrieval": result.retrieval.model_dump(mode="json"),
            "evaluation": result.evaluation.model_dump(mode="json"),
        },
        "metrics": dict(result.metrics),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    temporary_path.replace(path)
