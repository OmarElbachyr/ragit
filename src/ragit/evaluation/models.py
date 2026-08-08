"""Normalized models for page-level RAGit evaluation."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictInt


class QueryEvaluation(BaseModel):
    """Metric values for one canonical query."""

    model_config = ConfigDict(extra="forbid")

    query_id: StrictInt
    metrics: dict[str, float]


class EvaluationResult(BaseModel):
    """One persisted page-level evaluation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    aggregate: dict[str, float]
    per_query: list[QueryEvaluation]
    config_hash: str
    output_path: Path
    results_table: str
