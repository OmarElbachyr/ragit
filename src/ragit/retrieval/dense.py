"""Page-level retrieval with dense or late-interaction chunk scoring."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import Any, Literal

import numpy as np

from ragit.data.models import Collection, Query
from ragit.indexing.models import (
    DenseIndexResult,
    IndexResult,
    LateInteractionIndexResult,
)
from ragit.retrieval.models import RetrievalResult, RetrievedChunk, RetrievedPage
from ragit.retrieval.storage import save_retrieval_result


AggregationName = Literal["max", "sum", "mean"]
Aggregation = AggregationName | Callable[[list[RetrievedChunk]], float]

_LATE_INTERACTION_CANDIDATE_K = 500


class RetrievalConfigurationError(ValueError):
    """Raised when a retrieval request is invalid."""


def retrieve(
    *,
    collection: Collection,
    index_result: IndexResult,
    top_k: int = 100,
    aggregation: Aggregation = "max",
) -> RetrievalResult:
    """Retrieve ranked pages for every collection query.

    Chunks are the internal retrieval unit for both dense and late-interaction
    indexes. ``top_k`` is applied only after chunk-to-page aggregation.
    """
    queries = _validate_queries(collection.queries)
    _validate_top_k(top_k)
    aggregation_function = _resolve_aggregation(aggregation)

    if isinstance(index_result, DenseIndexResult):
        chunk_results_by_query = _retrieve_dense_chunks(
            queries=queries,
            index_result=index_result,
        )
    elif isinstance(index_result, LateInteractionIndexResult):
        chunk_results_by_query = _retrieve_late_interaction_chunks(
            queries=queries,
            index_result=index_result,
        )
    else:
        raise TypeError(
            "index_result must be a DenseIndexResult or "
            "LateInteractionIndexResult instance."
        )

    page_results: list[RetrievedPage] = []
    for query in queries:
        page_results.extend(
            _aggregate_query_pages(
                query=query,
                chunk_results=chunk_results_by_query[query.query_id],
                top_k=top_k,
                aggregation_function=aggregation_function,
            )
        )

    return save_retrieval_result(
        pdfs_path=collection.pdfs_path,
        index_result=index_result,
        queries=queries,
        top_k=top_k,
        aggregation=aggregation,
        results=page_results,
    )


# ---------------------------------------------------------------------------
# Dense chunk retrieval
# ---------------------------------------------------------------------------


def _retrieve_dense_chunks(
    *,
    queries: list[Query],
    index_result: DenseIndexResult,
) -> dict[int, list[RetrievedChunk]]:
    _validate_dense_index_result(index_result)
    num_chunks = int(index_result.index.ntotal)
    if num_chunks <= 0:
        raise RetrievalConfigurationError(
            "Dense retrieval requires an index containing at least one chunk."
        )

    encoder = _load_encoder(index_result)
    query_embeddings = _encode_queries(
        encoder=encoder,
        queries=queries,
        index_result=index_result,
    )

    try:
        scores, positions = index_result.index.search(query_embeddings, num_chunks)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RetrievalConfigurationError(
            "FAISS search failed for the dense query embeddings."
        ) from error

    scores = np.asarray(scores)
    positions = np.asarray(positions)
    expected_shape = (len(queries), num_chunks)
    if scores.shape != expected_shape or positions.shape != expected_shape:
        raise RetrievalConfigurationError(
            "FAISS search returned an unexpected result shape. Expected "
            f"{expected_shape}, got scores={scores.shape}, "
            f"positions={positions.shape}."
        )

    position_to_chunk_id = _position_mapping(index_result)
    chunks_by_id = _chunk_lookup(index_result)

    return {
        query.query_id: _normalize_dense_query_results(
            query=query,
            scores=scores[query_offset],
            positions=positions[query_offset],
            position_to_chunk_id=position_to_chunk_id,
            chunks_by_id=chunks_by_id,
        )
        for query_offset, query in enumerate(queries)
    }


def _normalize_dense_query_results(
    *,
    query: Query,
    scores: np.ndarray,
    positions: np.ndarray,
    position_to_chunk_id: dict[int, str],
    chunks_by_id: dict[str, Any],
) -> list[RetrievedChunk]:
    results: list[RetrievedChunk] = []
    for rank_offset, (raw_score, raw_position) in enumerate(
        zip(scores, positions, strict=True)
    ):
        position = int(raw_position)
        if position < 0 or position not in position_to_chunk_id:
            raise RetrievalConfigurationError(
                "FAISS returned an invalid index position "
                f"{position} for query_id={query.query_id}."
            )

        chunk_id = position_to_chunk_id[position]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise RetrievalConfigurationError(
                f"Dense index references unknown chunk_id={chunk_id!r}."
            )

        score = float(raw_score)
        if not np.isfinite(score):
            raise RetrievalConfigurationError(
                "FAISS returned a non-finite chunk score for "
                f"query_id={query.query_id}, chunk_id={chunk_id!r}."
            )

        results.append(
            RetrievedChunk(
                query_id=query.query_id,
                chunk_id=chunk_id,
                page_ids=list(chunk.page_ids),
                score=score,
                rank=rank_offset + 1,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Late-interaction chunk retrieval
# ---------------------------------------------------------------------------


def _retrieve_late_interaction_chunks(
    *,
    queries: list[Query],
    index_result: LateInteractionIndexResult,
) -> dict[int, list[RetrievedChunk]]:
    _validate_late_interaction_index_result(index_result)
    num_chunks = len(index_result.chunk_ids)
    if num_chunks <= 0:
        raise RetrievalConfigurationError(
            "Late-interaction retrieval requires an index containing at "
            "least one chunk."
        )

    encoder = _load_late_interaction_encoder(index_result)
    query_embeddings = _encode_late_interaction_queries(
        encoder=encoder,
        queries=queries,
        index_result=index_result,
    )
    candidate_k = min(_LATE_INTERACTION_CANDIDATE_K, num_chunks)
    try:
        raw_results = index_result.index.search(
            queries_embeddings=query_embeddings,
            top_k=candidate_k,
            show_progress=False,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise RetrievalConfigurationError(
            "FastPlaid search failed for the late-interaction query embeddings."
        ) from error

    try:
        result_rows = list(raw_results)
    except TypeError as error:
        raise RetrievalConfigurationError(
            "FastPlaid returned an invalid retrieval result."
        ) from error

    if len(result_rows) != len(queries):
        raise RetrievalConfigurationError(
            "FastPlaid returned an unexpected number of query result lists."
        )

    chunks_by_id = _late_interaction_chunk_lookup(index_result)
    normalized: dict[int, list[RetrievedChunk]] = {}
    for query, row in zip(queries, result_rows, strict=True):
        row = list(row)
        normalized[query.query_id] = _normalize_fast_plaid_query_results(
            query=query,
            row=row,
            chunks_by_id=chunks_by_id,
            chunk_ids=index_result.chunk_ids,
        )
    return normalized


def _normalize_fast_plaid_query_results(
    *,
    query: Query,
    row: list[Any],
    chunks_by_id: dict[str, Any],
    chunk_ids: list[str],
) -> list[RetrievedChunk]:
    results: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    for rank_offset, item in enumerate(row):
        try:
            index_position, raw_score = item
            chunk_id = chunk_ids[int(index_position)]
            score = float(raw_score)
        except (IndexError, TypeError, ValueError) as error:
            raise RetrievalConfigurationError(
                "FastPlaid result items must contain a valid index position "
                "and numeric score."
            ) from error

        if chunk_id in seen_ids:
            raise RetrievalConfigurationError(
                f"FastPlaid returned duplicate chunk_id={chunk_id!r}."
            )
        if not np.isfinite(score):
            raise RetrievalConfigurationError(
                "FastPlaid returned a non-finite chunk score for "
                f"query_id={query.query_id}, chunk_id={chunk_id!r}."
            )

        chunk = chunks_by_id[chunk_id]
        seen_ids.add(chunk_id)
        results.append(
            RetrievedChunk(
                query_id=query.query_id,
                chunk_id=chunk_id,
                page_ids=list(chunk.page_ids),
                score=score,
                rank=rank_offset + 1,
            )
        )

    return results


def _encode_late_interaction_queries(
    *,
    encoder: Any,
    queries: list[Query],
    index_result: LateInteractionIndexResult,
) -> Any:
    if index_result.config is None:
        raise RetrievalConfigurationError(
            "Late-interaction index result is missing its configuration."
        )
    options = index_result.config.options
    try:
        return encoder.encode_query(
            [query.text for query in queries],
            batch_size=options.get("batch_size", 32),
            show_progress_bar=options.get("show_progress_bar", False),
        )
    except TypeError as error:
        raise RetrievalConfigurationError(
            f"Model {index_result.config.model_name!r} does not provide a "
            "compatible Sentence Transformers MultiVectorEncoder query "
            "interface."
        ) from error


def _load_late_interaction_encoder(
    index_result: LateInteractionIndexResult,
) -> Any:
    if index_result.config is None:
        raise RetrievalConfigurationError(
            "Late-interaction index result is missing its configuration."
        )
    try:
        from sentence_transformers import MultiVectorEncoder
    except ImportError as error:
        raise ImportError(
            "sentence-transformers>=6 is required for late-interaction "
            "RAGit retrieval."
        ) from error

    kwargs: dict[str, Any] = {}
    device = index_result.config.options.get("device")
    if device is not None:
        kwargs["device"] = device
    if index_result.config.options.get("trust_remote_code", False):
        kwargs["trust_remote_code"] = True
    return MultiVectorEncoder(index_result.config.model_name, **kwargs)


# ---------------------------------------------------------------------------
# Common page aggregation
# ---------------------------------------------------------------------------


def _aggregate_query_pages(
    *,
    query: Query,
    chunk_results: list[RetrievedChunk],
    top_k: int,
    aggregation_function: Callable[[list[RetrievedChunk]], float],
) -> list[RetrievedPage]:
    chunks_by_page: dict[int, list[RetrievedChunk]] = {}
    for chunk_result in chunk_results:
        for page_id in chunk_result.page_ids:
            chunks_by_page.setdefault(page_id, []).append(chunk_result)

    scored_pages: list[tuple[int, float, list[RetrievedChunk]]] = []
    for page_id, contributors in chunks_by_page.items():
        try:
            raw_page_score = aggregation_function(list(contributors))
        except RetrievalConfigurationError:
            raise
        except Exception as error:
            raise RetrievalConfigurationError(
                "Custom aggregation failed for "
                f"query_id={query.query_id}, page_id={page_id}."
            ) from error

        page_score = _normalize_page_score(
            raw_page_score,
            query_id=query.query_id,
            page_id=page_id,
        )
        scored_pages.append((page_id, page_score, contributors))

    scored_pages.sort(key=lambda item: (-item[1], item[0]))
    selected_pages = scored_pages[:top_k]
    return [
        RetrievedPage(
            query_id=query.query_id,
            page_id=page_id,
            score=page_score,
            rank=rank_offset + 1,
            chunks=list(contributors),
        )
        for rank_offset, (page_id, page_score, contributors) in enumerate(
            selected_pages
        )
    ]


def _resolve_aggregation(
    aggregation: Aggregation,
) -> Callable[[list[RetrievedChunk]], float]:
    if aggregation == "max":
        return lambda chunks: max(chunk.score for chunk in chunks)
    if aggregation == "sum":
        return lambda chunks: sum(chunk.score for chunk in chunks)
    if aggregation == "mean":
        return lambda chunks: sum(chunk.score for chunk in chunks) / len(chunks)
    if callable(aggregation):
        return aggregation
    raise RetrievalConfigurationError(
        "aggregation must be one of {'max', 'sum', 'mean'} or a callable."
    )


def _normalize_page_score(
    score: Any,
    *,
    query_id: int,
    page_id: int,
) -> float:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise RetrievalConfigurationError(
            "Aggregation must return one numeric page score for "
            f"query_id={query_id}, page_id={page_id}."
        )
    normalized = float(score)
    if not np.isfinite(normalized):
        raise RetrievalConfigurationError(
            "Aggregation returned a non-finite page score for "
            f"query_id={query_id}, page_id={page_id}."
        )
    return normalized


# ---------------------------------------------------------------------------
# Existing dense helpers
# ---------------------------------------------------------------------------


def _encode_queries(
    *,
    encoder: Any,
    queries: list[Query],
    index_result: DenseIndexResult,
) -> np.ndarray:
    if index_result.config is None:
        raise RetrievalConfigurationError(
            "Dense index result is missing its indexing configuration."
        )

    texts = [query.text for query in queries]
    options = index_result.config.options
    encode_kwargs = {
        "batch_size": options.get("batch_size", 32),
        "show_progress_bar": options.get("show_progress_bar", False),
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }
    encode_query = getattr(encoder, "encode_query", None)
    encode = getattr(encoder, "encode", None)

    try:
        if callable(encode_query):
            raw_embeddings = encode_query(texts, **encode_kwargs)
        elif callable(encode):
            raw_embeddings = encode(texts, **encode_kwargs)
        else:
            raise RetrievalConfigurationError(
                f"Model {index_result.config.model_name!r} does not provide "
                "a compatible query encoding method."
            )
    except RetrievalConfigurationError:
        raise
    except TypeError as error:
        raise RetrievalConfigurationError(
            f"Model {index_result.config.model_name!r} does not provide a "
            "compatible query encoding interface."
        ) from error

    try:
        embeddings = np.asarray(raw_embeddings)
    except (TypeError, ValueError) as error:
        raise RetrievalConfigurationError(
            "Query encoder did not return a rectangular dense embedding matrix."
        ) from error

    expected_dimension = int(index_result.index.d)
    if embeddings.ndim != 2 or embeddings.shape != (
        len(queries),
        expected_dimension,
    ):
        raise RetrievalConfigurationError(
            "Query encoder returned an incompatible embedding shape. "
            f"Expected ({len(queries)}, {expected_dimension}), got "
            f"{embeddings.shape}."
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RetrievalConfigurationError(
            "Query encoder produced at least one zero vector, which cannot "
            "be L2-normalized."
        )
    embeddings /= norms
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def _load_encoder(index_result: DenseIndexResult) -> Any:
    if index_result.config is None:
        raise RetrievalConfigurationError(
            "Dense index result is missing its indexing configuration."
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for dense RAGit retrieval."
        ) from error

    kwargs: dict[str, Any] = {}
    device = index_result.config.options.get("device")
    if device is not None:
        kwargs["device"] = device
    return SentenceTransformer(index_result.config.model_name, **kwargs)


def _validate_queries(queries: list[Query]) -> list[Query]:
    if not queries:
        raise RetrievalConfigurationError(
            "Retrieval requires at least one collection query."
        )

    seen_ids: set[int] = set()
    normalized: list[Query] = []
    for query in queries:
        query_id = getattr(query, "query_id", None)
        text = getattr(query, "text", None)
        if isinstance(query_id, bool) or not isinstance(query_id, int):
            raise RetrievalConfigurationError(
                "Collection queries must have valid integer query_id values."
            )
        if query_id in seen_ids:
            raise RetrievalConfigurationError(
                f"Duplicate query_id in collection: {query_id}."
            )
        if not isinstance(text, str) or not text:
            raise RetrievalConfigurationError(
                f"Query {query_id} must contain non-empty text."
            )
        seen_ids.add(query_id)
        normalized.append(query)
    return normalized


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RetrievalConfigurationError("top_k must be an integer.")
    if top_k <= 0:
        raise RetrievalConfigurationError("top_k must be greater than zero.")


def _validate_dense_index_result(index_result: DenseIndexResult) -> None:
    if index_result.config is None:
        raise RetrievalConfigurationError(
            "Dense index result is missing its indexing configuration."
        )
    if index_result.config.index_type != "dense":
        raise RetrievalConfigurationError(
            "DenseIndexResult requires index_type='dense'."
        )
    if not callable(getattr(index_result.index, "search", None)):
        raise RetrievalConfigurationError(
            "Dense index result does not expose a FAISS-compatible search()."
        )
    if not hasattr(index_result.index, "ntotal") or not hasattr(
        index_result.index,
        "d",
    ):
        raise RetrievalConfigurationError(
            "Dense index result is missing FAISS size/dimension metadata."
        )
    if len(index_result.chunk_mapping) != int(index_result.index.ntotal):
        raise RetrievalConfigurationError(
            "Dense index chunk mapping size does not match FAISS index size."
        )


def _validate_late_interaction_index_result(
    index_result: LateInteractionIndexResult,
) -> None:
    if index_result.config is None:
        raise RetrievalConfigurationError(
            "Late-interaction index result is missing its configuration."
        )
    if index_result.config.index_type != "late_interaction":
        raise RetrievalConfigurationError(
            "LateInteractionIndexResult requires "
            "index_type='late_interaction'."
        )
    if not index_result.chunk_ids:
        raise RetrievalConfigurationError(
            "Late-interaction index result contains no chunk IDs."
        )
    if len(index_result.chunk_ids) != len(index_result.chunks):
        raise RetrievalConfigurationError(
            "Late-interaction chunk ID count does not match originating "
            "chunk provenance."
        )


def _position_mapping(index_result: DenseIndexResult) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in index_result.chunk_mapping:
        if item.index_position in mapping:
            raise RetrievalConfigurationError(
                f"Duplicate FAISS index position: {item.index_position}."
            )
        mapping[item.index_position] = item.chunk_id
    return mapping


def _chunk_lookup(index_result: DenseIndexResult) -> dict[str, Any]:
    chunks_by_id: dict[str, Any] = {}
    for chunk in index_result.chunks:
        if chunk.chunk_id in chunks_by_id:
            raise RetrievalConfigurationError(
                f"Duplicate originating chunk_id: {chunk.chunk_id!r}."
            )
        chunks_by_id[chunk.chunk_id] = chunk

    indexed_ids = {item.chunk_id for item in index_result.chunk_mapping}
    missing = sorted(indexed_ids - set(chunks_by_id))
    if missing:
        raise RetrievalConfigurationError(
            "Dense index result is missing originating chunk provenance for "
            f"chunk IDs: {missing[:5]}."
        )
    return chunks_by_id


def _late_interaction_chunk_lookup(
    index_result: LateInteractionIndexResult,
) -> dict[str, Any]:
    chunks_by_id: dict[str, Any] = {}
    for chunk in index_result.chunks:
        if chunk.chunk_id in chunks_by_id:
            raise RetrievalConfigurationError(
                f"Duplicate originating chunk_id: {chunk.chunk_id!r}."
            )
        chunks_by_id[chunk.chunk_id] = chunk

    missing = sorted(set(index_result.chunk_ids) - set(chunks_by_id))
    if missing:
        raise RetrievalConfigurationError(
            "Late-interaction index result is missing originating chunk "
            f"provenance for chunk IDs: {missing[:5]}."
        )
    return chunks_by_id
