"""Configuration and result models for RAGit experiments."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ragit.chunking import ChunkingConfig
from ragit.evaluation import EvaluationConfig
from ragit.indexing import IndexingConfig
from ragit.parsing import ParsingConfig
from ragit.retrieval import RetrievalConfig


class ExperimentConfig(BaseModel):
    """Configuration for one complete RAGit experiment."""

    model_config = ConfigDict(extra="forbid")

    parsing: ParsingConfig
    chunking: ChunkingConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig


class ArtifactReference(BaseModel):
    """Lightweight reference to one persisted stage artifact."""

    model_config = ConfigDict(extra="forbid")

    config_hash: str
    output_path: Path


class ExperimentResult(BaseModel):
    """Lightweight result for one completed experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiment_hash: str
    config: ExperimentConfig
    parsing: ArtifactReference
    chunking: ArtifactReference
    indexing: ArtifactReference
    retrieval: ArtifactReference
    evaluation: ArtifactReference
    metrics: dict[str, float] = Field(default_factory=dict)
    output_path: Path
