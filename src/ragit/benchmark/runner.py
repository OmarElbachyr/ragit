"""Explicit multi-experiment benchmark orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ragit.benchmark.models import BenchmarkExperimentResult, BenchmarkResult
from ragit.benchmark.storage import (
    benchmark_configuration_hash,
    benchmark_output_path,
    format_benchmark_results_table,
    save_benchmark_result,
    _shared_evaluation,
    _validated_experiment_items,
)
from ragit.experiment import ExperimentConfig, run_experiment

if TYPE_CHECKING:
    from ragit.data.models import Collection


class BenchmarkExecutionError(RuntimeError):
    """Raised when one labeled experiment fails during a benchmark."""


def run_benchmark(
    *,
    collection: "Collection",
    experiments: Mapping[str, ExperimentConfig],
    overwrite_parsing: bool = False,
) -> BenchmarkResult:
    """Run explicitly defined experiments sequentially and compare aggregates."""
    items = _validated_experiment_items(experiments)
    evaluation = _shared_evaluation(items)

    completed: list[BenchmarkExperimentResult] = []
    for label, config in items:
        try:
            experiment_result = run_experiment(
                collection=collection,
                config=config,
                overwrite_parsing=overwrite_parsing,
            )
        except Exception as error:
            raise BenchmarkExecutionError(
                f"Benchmark experiment {label!r} failed."
            ) from error

        completed.append(
            BenchmarkExperimentResult(
                label=label,
                experiment_hash=experiment_result.experiment_hash,
                metrics=dict(experiment_result.metrics),
                output_path=experiment_result.output_path,
            )
        )

    benchmark_hash = benchmark_configuration_hash(experiments)
    results_table = format_benchmark_results_table(
        experiments=completed,
        evaluation=evaluation,
    )
    result = BenchmarkResult(
        benchmark_hash=benchmark_hash,
        experiments=completed,
        evaluation=evaluation,
        output_path=benchmark_output_path(collection.pdfs_path, experiments),
        results_table=results_table,
    )
    return save_benchmark_result(
        pdfs_path=collection.pdfs_path,
        experiments=experiments,
        result=result,
    )
