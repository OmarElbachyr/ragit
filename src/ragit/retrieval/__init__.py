"""Public retrieval interfaces."""

from ragit.retrieval.dense import (
    Aggregation,
    AggregationName,
    RetrievalConfigurationError,
    retrieve,
)
from ragit.retrieval.models import RetrievalResult, RetrievedChunk, RetrievedPage
from ragit.retrieval.storage import (
    RetrievalStorageError,
    aggregation_identifier,
    query_set_hash,
    retrieval_configuration_hash,
)

__all__ = [
    "Aggregation",
    "AggregationName",
    "RetrievalConfigurationError",
    "RetrievalResult",
    "RetrievalStorageError",
    "RetrievedChunk",
    "RetrievedPage",
    "aggregation_identifier",
    "query_set_hash",
    "retrieval_configuration_hash",
    "retrieve",
]
