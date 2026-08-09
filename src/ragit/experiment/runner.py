"""Single-experiment orchestration over existing RAGit pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragit.chunking import chunk_document, save_chunking_result
from ragit.evaluation import evaluate
from ragit.experiment.models import (
    ArtifactReference,
    ExperimentConfig,
    ExperimentResult,
)
from ragit.experiment.storage import (
    experiment_configuration_hash,
    experiment_output_path,
    save_experiment_result,
)
from ragit.indexing import build_index
from ragit.parsing import parse_document
from ragit.retrieval import retrieve

if TYPE_CHECKING:
    from ragit.data.models import Collection


def run_experiment(
    *,
    collection: "Collection",
    config: ExperimentConfig,
    overwrite_parsing: bool = False,
) -> ExperimentResult:
    """Run one complete experiment by coordinating existing public APIs.

    Set ``overwrite_parsing=True`` to rebuild cached parsing results for
    every document.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig instance.")

    documents = list(collection.documents)
    if not documents:
        raise ValueError("Experiment requires a collection with at least one document.")

    parsing_results = []
    parsing_config_hash = None

    for document in documents:
        current_parsing_result = parse_document(
            collection=collection,
            doc_id=document.doc_id,
            config=config.parsing,
            overwrite=overwrite_parsing,
        )
        current_hash = _require_non_empty_string(
            current_parsing_result,
            "config_hash",
            stage="parsing",
        )
        if parsing_config_hash is None:
            parsing_config_hash = current_hash
        elif current_hash != parsing_config_hash:
            raise ValueError(
                "Parsing returned inconsistent configuration hashes across "
                "documents in the same experiment."
            )
        parsing_results.append(current_parsing_result)

    assert parsing_config_hash is not None

    chunks = []
    for current_parsing_result in parsing_results:
        chunks.extend(
            chunk_document(
                parsed_pages=current_parsing_result.pages,
                config=config.chunking,
            )
        )

    parsing_result = parsing_results[0]

    chunking_result = save_chunking_result(
        pdfs_path=collection.pdfs_path,
        config=config.chunking,
        parsing_config_hash=parsing_config_hash,
        chunks=chunks,
        replace=True,
    )

    index_result = build_index(
        collection=collection,
        chunking_result=chunking_result,
        config=config.indexing,
    )

    retrieval_result = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=config.retrieval.top_k,
        aggregation=config.retrieval.aggregation,
    )

    evaluation_result = evaluate(
        collection=collection,
        retrieval_result=retrieval_result,
        metrics=config.evaluation.metrics,
        k=config.evaluation.k,
    )

    experiment_hash = experiment_configuration_hash(config)
    result = ExperimentResult(
        experiment_hash=experiment_hash,
        config=config,
        parsing=_artifact_reference(parsing_result, stage="parsing"),
        chunking=_artifact_reference(chunking_result, stage="chunking"),
        indexing=_artifact_reference(index_result, stage="indexing"),
        retrieval=_artifact_reference(retrieval_result, stage="retrieval"),
        evaluation=_artifact_reference(evaluation_result, stage="evaluation"),
        metrics=dict(evaluation_result.aggregate),
        output_path=experiment_output_path(collection.pdfs_path, config),
    )
    return save_experiment_result(
        pdfs_path=collection.pdfs_path,
        result=result,
    )


def _artifact_reference(result: Any, *, stage: str) -> ArtifactReference:
    """Extract the common lightweight provenance from one stage result."""
    config_hash = _require_non_empty_string(result, "config_hash", stage=stage)
    output_path = getattr(result, "output_path", None)
    if output_path is None:
        raise ValueError(f"{stage.capitalize()} result has no output_path.")
    return ArtifactReference(
        config_hash=config_hash,
        output_path=output_path,
    )


def _require_non_empty_string(result: Any, field: str, *, stage: str) -> str:
    value = getattr(result, field, None)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{stage.capitalize()} result has no valid {field}."
        )
    return value
