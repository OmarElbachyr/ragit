"""Deterministic identity and lightweight persistence for benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tabulate import tabulate

from ragit.evaluation import EvaluationConfig
from ragit.experiment import ExperimentConfig, experiment_configuration_hash
from ragit.benchmark.models import BenchmarkExperimentResult, BenchmarkResult


CONFIG_FILE = "config.json"
RESULTS_FILE = "results.json"
RESULTS_TABLE_FILE = "results.txt"


class BenchmarkStorageError(ValueError):
    """Raised when benchmark persistence data is inconsistent."""


def benchmark_configuration_hash(
    experiments: Mapping[str, ExperimentConfig],
) -> str:
    """Return a deterministic hash of the ordered, labeled experiment set."""
    items = _validated_experiment_items(experiments)
    evaluation = _shared_evaluation(items)
    payload = _configuration_payload(items=items, evaluation=evaluation)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def benchmark_output_path(
    pdfs_path: str | Path,
    experiments: Mapping[str, ExperimentConfig],
) -> Path:
    """Return the deterministic output directory for one benchmark."""
    return (
        Path(pdfs_path)
        / ".ragit"
        / "benchmarks"
        / benchmark_configuration_hash(experiments)
    )


def save_benchmark_result(
    *,
    pdfs_path: str | Path,
    experiments: Mapping[str, ExperimentConfig],
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Persist the latest lightweight benchmark comparison artifacts."""
    items = _validated_experiment_items(experiments)
    evaluation = _shared_evaluation(items)
    expected_hash = benchmark_configuration_hash(experiments)
    if result.benchmark_hash != expected_hash:
        raise BenchmarkStorageError(
            "BenchmarkResult benchmark_hash does not match its experiments."
        )

    output_path = benchmark_output_path(pdfs_path, experiments)
    if result.output_path != output_path or result.evaluation != evaluation:
        result = result.model_copy(
            update={"output_path": output_path, "evaluation": evaluation}
        )

    output_path.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        output_path / CONFIG_FILE,
        _configuration_payload(items=items, evaluation=evaluation),
    )
    _write_json_atomic(
        output_path / RESULTS_FILE,
        _results_payload(result.experiments),
    )
    _write_text_atomic(
        output_path / RESULTS_TABLE_FILE,
        result.results_table + "\n",
    )
    return result


def format_benchmark_results_table(
    *,
    experiments: Sequence[BenchmarkExperimentResult],
    evaluation: EvaluationConfig,
) -> str:
    """Format benchmark metrics using grouped metric-family headers."""

    experiment_width = max(
        len("Experiment"),
        *(len(experiment.label) for experiment in experiments),
    )
    value_width = 8

    # Validate expected aggregate metrics first.
    for experiment in experiments:
        for metric in evaluation.metrics:
            for cutoff in evaluation.k:
                metric_label = f"{metric}@{cutoff}"
                if metric_label not in experiment.metrics:
                    raise BenchmarkStorageError(
                        f"Experiment {experiment.label!r} is missing aggregate "
                        f"metric {metric_label!r}."
                    )

    # Width occupied by one complete metric family.
    group_width = (
        len(evaluation.k) * value_width
        + (len(evaluation.k) - 1)
    )

    # First header: metric families.
    header_1 = (
        " " * experiment_width
        + " | "
        + " | ".join(
            metric.center(group_width)
            for metric in evaluation.metrics
        )
    )

    # Second header: cutoffs.
    cutoff_group = " ".join(
        f"@{cutoff}".center(value_width)
        for cutoff in evaluation.k
    )

    header_2 = (
        "Experiment".ljust(experiment_width)
        + " | "
        + " | ".join(
            cutoff_group
            for _ in evaluation.metrics
        )
    )

    separator = (
        "-" * experiment_width
        + "-+-"
        + "-+-".join(
            "-" * group_width
            for _ in evaluation.metrics
        )
    )

    rows: list[str] = []

    for experiment in experiments:
        metric_groups: list[str] = []

        for metric in evaluation.metrics:
            values = [
                experiment.metrics[f"{metric}@{cutoff}"]
                for cutoff in evaluation.k
            ]

            metric_groups.append(
                " ".join(
                    f"{value:.4f}".rjust(value_width)
                    for value in values
                )
            )

        rows.append(
            experiment.label.ljust(experiment_width)
            + " | "
            + " | ".join(metric_groups)
        )

    return "\n".join(
        [
            header_1,
            header_2,
            separator,
            *rows,
        ]
    )


def _configuration_payload(
    *,
    items: Sequence[tuple[str, ExperimentConfig]],
    evaluation: EvaluationConfig,
) -> dict[str, Any]:
    return {
        "experiments": [
            {
                "label": label,
                "experiment_hash": experiment_configuration_hash(config),
            }
            for label, config in items
        ],
        "evaluation": evaluation.model_dump(mode="json"),
    }


def _results_payload(
    experiments: Sequence[BenchmarkExperimentResult],
) -> dict[str, Any]:
    return {
        "experiments": [
            {
                "label": experiment.label,
                "experiment_hash": experiment.experiment_hash,
                "metrics": dict(experiment.metrics),
            }
            for experiment in experiments
        ]
    }


def _validated_experiment_items(
    experiments: Mapping[str, ExperimentConfig],
) -> list[tuple[str, ExperimentConfig]]:
    if not isinstance(experiments, Mapping):
        raise TypeError("experiments must be a mapping of label to ExperimentConfig.")
    if not experiments:
        raise ValueError("Benchmark requires at least one experiment.")

    items = list(experiments.items())
    for label, config in items:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Benchmark experiment labels must be non-empty strings.")
        if not isinstance(config, ExperimentConfig):
            raise TypeError(
                f"Benchmark experiment {label!r} must be an ExperimentConfig instance."
            )
    return items


def _shared_evaluation(
    items: Sequence[tuple[str, ExperimentConfig]],
) -> EvaluationConfig:
    reference_label, reference_config = items[0]
    reference = reference_config.evaluation

    for label, config in items[1:]:
        current = config.evaluation
        if current.metrics != reference.metrics:
            raise ValueError(
                "All benchmark experiments must use the same evaluation metric "
                f"families. {reference_label!r} uses {reference.metrics}, while "
                f"{label!r} uses {current.metrics}."
            )
        if current.k != reference.k:
            raise ValueError(
                "All benchmark experiments must use the same evaluation k values. "
                f"{reference_label!r} uses {reference.k}, while "
                f"{label!r} uses {current.k}."
            )
    return reference


def _metric_labels(evaluation: EvaluationConfig) -> list[str]:
    return [
        f"{metric}@{cutoff}"
        for metric in evaluation.metrics
        for cutoff in evaluation.k
    ]


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
