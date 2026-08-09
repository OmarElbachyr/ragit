"""Configuration-free result models for RAGit benchmarks."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ragit.evaluation import EvaluationConfig


class BenchmarkExperimentResult(BaseModel):
    """Lightweight benchmark reference to one completed experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    label: str
    experiment_hash: str
    metrics: dict[str, float] = Field(default_factory=dict)
    output_path: Path


class BenchmarkResult(BaseModel):
    """Lightweight aggregate comparison for one benchmark run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    benchmark_hash: str
    experiments: list[BenchmarkExperimentResult]
    evaluation: EvaluationConfig
    output_path: Path
    results_table: str
