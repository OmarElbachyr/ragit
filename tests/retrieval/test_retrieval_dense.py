"""Focused tests for dense page-level retrieval with chunk aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ragit.chunking import Chunk
from ragit.data.models import Query
from ragit.indexing import DenseIndexResult, IndexedChunk, IndexingConfig
from ragit.retrieval import (
    RetrievalConfigurationError,
    RetrievedChunk,
    retrieve,
)


class FakeQueryEncoder:
    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings)
        self.query_calls = 0
        self.encode_calls = 0
        self.last_inputs = None
        self.last_kwargs = None

    def encode_query(self, inputs, **kwargs):
        self.query_calls += 1
        self.last_inputs = list(inputs)
        self.last_kwargs = kwargs
        return self.embeddings.copy()

    def encode(self, inputs, **kwargs):
        self.encode_calls += 1
        return self.embeddings.copy()


class FakeGenericEncoder:
    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings)
        self.encode_calls = 0

    def encode(self, inputs, **kwargs):
        self.encode_calls += 1
        return self.embeddings.copy()


class FakeSearchIndex:
    def __init__(self, *, dimension, ntotal, scores, positions):
        self.d = dimension
        self.ntotal = ntotal
        self.scores = np.asarray(scores, dtype=np.float32)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.search_calls = 0
        self.last_queries = None
        self.last_k = None

    def search(self, queries, k):
        self.search_calls += 1
        self.last_queries = np.asarray(queries).copy()
        self.last_k = k
        return self.scores[:, :k], self.positions[:, :k]


@pytest.fixture
def chunks():
    return [
        Chunk(
            chunk_id="doc-a:00000000",
            doc_id="doc-a",
            page_ids=[10],
            content="alpha",
        ),
        Chunk(
            chunk_id="doc-a:00000001",
            doc_id="doc-a",
            page_ids=[10, 11],
            content="beta",
        ),
        Chunk(
            chunk_id="doc-a:00000002",
            doc_id="doc-a",
            page_ids=[12],
            content="gamma",
        ),
    ]


def _collection(tmp_path: Path, queries):
    return SimpleNamespace(pdfs_path=tmp_path, queries=queries)


def _index_result(tmp_path: Path, chunks, index):
    return DenseIndexResult(
        index=index,
        chunk_mapping=[
            IndexedChunk(index_position=i, chunk_id=chunk.chunk_id)
            for i, chunk in enumerate(chunks)
        ],
        config_hash="dense-index-hash",
        output_path=tmp_path / ".ragit" / "indexing" / "dense" / "x",
        config=IndexingConfig(
            model_name="fake/dense",
            options={"batch_size": 8},
        ),
        chunks=chunks,
    )


def _patch_encoder(monkeypatch, encoder):
    import ragit.retrieval.dense as dense_module

    monkeypatch.setattr(dense_module, "_load_encoder", lambda result: encoder)


def test_retrieval_scores_all_chunks_then_returns_top_k_pages(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(
        tmp_path,
        [Query(query_id=7, text="first question")],
    )
    encoder = FakeQueryEncoder([[3.0, 4.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.7]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    result = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=2,
    )

    assert encoder.query_calls == 1
    assert encoder.encode_calls == 0
    assert encoder.last_inputs == ["first question"]
    assert encoder.last_kwargs["convert_to_numpy"] is True
    assert encoder.last_kwargs["normalize_embeddings"] is False

    assert index.search_calls == 1
    assert index.last_k == 3
    assert index.last_queries.dtype == np.float32
    assert np.allclose(index.last_queries[0], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(index.last_queries, axis=1), [1.0])

    assert [(page.page_id, page.rank) for page in result.results] == [
        (10, 1),
        (11, 2),
    ]
    assert [page.score for page in result.results] == pytest.approx([0.9, 0.8])

    page_10 = result.results[0]
    assert [chunk.chunk_id for chunk in page_10.chunks] == [
        "doc-a:00000000",
        "doc-a:00000001",
    ]
    assert [chunk.rank for chunk in page_10.chunks] == [1, 2]

    page_11 = result.results[1]
    assert [chunk.chunk_id for chunk in page_11.chunks] == [
        "doc-a:00000001"
    ]
    assert page_11.chunks[0].page_ids == [10, 11]


def test_generic_encode_is_fallback_when_encode_query_is_unavailable(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeGenericEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[1.0, 0.5, 0.1]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    retrieve(collection=collection, index_result=index_result, top_k=1)
    assert encoder.encode_calls == 1


@pytest.mark.parametrize(
    ("aggregation", "expected_scores"),
    [
        ("max", {10: 0.9, 11: 0.8, 12: 0.3}),
        ("sum", {10: 1.7, 11: 0.8, 12: 0.3}),
        ("mean", {10: 0.85, 11: 0.8, 12: 0.3}),
    ],
)
def test_builtin_aggregation_methods(
    tmp_path,
    monkeypatch,
    chunks,
    aggregation,
    expected_scores,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.3]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    result = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=100,
        aggregation=aggregation,
    )

    scores = {page.page_id: page.score for page in result.results}
    assert scores == pytest.approx(expected_scores)


def test_custom_aggregation_receives_full_chunk_records(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.3]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    observed = []

    def custom_aggregation(chunk_results):
        observed.append(list(chunk_results))
        assert all(isinstance(item, RetrievedChunk) for item in chunk_results)
        return sum(item.score / item.rank for item in chunk_results)

    result = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=100,
        aggregation=custom_aggregation,
    )

    assert observed
    page_10 = next(page for page in result.results if page.page_id == 10)
    assert page_10.score == pytest.approx(0.9 + 0.8 / 2)
    assert [chunk.rank for chunk in page_10.chunks] == [1, 2]
    assert [chunk.page_ids for chunk in page_10.chunks] == [[10], [10, 11]]


def test_multi_page_chunk_contributes_to_each_page_and_keeps_chunk_rank(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.95, 0.9, 0.1]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    result = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=3,
        aggregation="max",
    )

    page_10 = next(page for page in result.results if page.page_id == 10)
    page_11 = next(page for page in result.results if page.page_id == 11)

    shared_10 = next(
        chunk for chunk in page_10.chunks if chunk.chunk_id == "doc-a:00000001"
    )
    shared_11 = next(
        chunk for chunk in page_11.chunks if chunk.chunk_id == "doc-a:00000001"
    )

    assert shared_10.rank == 2
    assert shared_11.rank == 2
    assert shared_10.page_ids == [10, 11]
    assert shared_11.page_ids == [10, 11]


def test_multiple_queries_have_independent_page_rankings(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(
        tmp_path,
        [
            Query(query_id=4, text="q1"),
            Query(query_id=9, text="q2"),
        ],
    )
    encoder = FakeQueryEncoder([[1.0, 0.0], [0.0, 2.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[
            [0.9, 0.7, 0.2],
            [0.95, 0.8, 0.4],
        ],
        positions=[
            [0, 2, 1],
            [1, 2, 0],
        ],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    result = retrieve(collection=collection, index_result=index_result, top_k=2)

    assert [(r.query_id, r.rank, r.page_id) for r in result.results] == [
        (4, 1, 10),
        (4, 2, 12),
        (9, 1, 10),
        (9, 2, 11),
    ]


def test_top_k_larger_than_number_of_pages_returns_all_pages(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.7]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    result = retrieve(collection=collection, index_result=index_result, top_k=100)

    assert index.last_k == 3
    assert {page.page_id for page in result.results} == {10, 11, 12}
    assert len(result.results) == 3


def test_invalid_faiss_position_is_rejected(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.7]],
        positions=[[0, -1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    with pytest.raises(RetrievalConfigurationError, match="invalid index position"):
        retrieve(collection=collection, index_result=index_result, top_k=1)


def test_results_json_contains_pages_and_all_contributing_chunks(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.3]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    first = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=2,
        aggregation="sum",
    )
    results_path = first.output_path / "results.json"
    config_path = first.output_path / "config.json"

    saved_results = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved_results["queries"][0]["query_id"] == 1
    pages = saved_results["queries"][0]["pages"]
    assert len(pages) == 2
    assert pages[0]["page_id"] == 10
    assert pages[0]["rank"] == 1
    assert pages[0]["score"] == pytest.approx(1.7)
    assert [chunk["chunk_id"] for chunk in pages[0]["chunks"]] == [
        "doc-a:00000000",
        "doc-a:00000001",
    ]
    assert [chunk["rank"] for chunk in pages[0]["chunks"]] == [1, 2]
    assert pages[0]["chunks"][1]["page_ids"] == [10, 11]

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["retrieval_type"] == "dense"
    assert saved_config["index_config_hash"] == "dense-index-hash"
    assert saved_config["model_name"] == "fake/dense"
    assert saved_config["top_k"] == 2
    assert saved_config["top_k_unit"] == "pages"
    assert saved_config["aggregation"] == "sum"
    assert saved_config["query_set_hash"]

    results_path.write_text("stale\n", encoding="utf-8")
    second = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=2,
        aggregation="sum",
    )

    assert first.output_path == second.output_path
    assert encoder.query_calls == 2
    assert index.search_calls == 2
    assert results_path.read_text(encoding="utf-8") != "stale\n"


def test_custom_aggregation_identifier_is_persisted(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.3]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    def custom_aggregation(chunk_results):
        return max(chunk.score for chunk in chunk_results)

    result = retrieve(
        collection=collection,
        index_result=index_result,
        top_k=2,
        aggregation=custom_aggregation,
    )

    config = json.loads(
        (result.output_path / "config.json").read_text(encoding="utf-8")
    )
    assert config["aggregation"].startswith("custom:")
    assert "custom_aggregation" in config["aggregation"]


def test_validation_requires_queries_positive_top_k_and_valid_aggregation(
    tmp_path,
    chunks,
):
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[1.0, 0.5, 0.2]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)

    with pytest.raises(RetrievalConfigurationError, match="at least one"):
        retrieve(
            collection=_collection(tmp_path, []),
            index_result=index_result,
            top_k=1,
        )

    duplicate_queries = [
        SimpleNamespace(query_id=1, text="a"),
        SimpleNamespace(query_id=1, text="b"),
    ]
    with pytest.raises(RetrievalConfigurationError, match="Duplicate query_id"):
        retrieve(
            collection=_collection(tmp_path, duplicate_queries),
            index_result=index_result,
            top_k=1,
        )

    with pytest.raises(RetrievalConfigurationError, match="greater than zero"):
        retrieve(
            collection=_collection(tmp_path, [Query(query_id=1, text="q")]),
            index_result=index_result,
            top_k=0,
        )

    with pytest.raises(RetrievalConfigurationError, match="aggregation"):
        retrieve(
            collection=_collection(tmp_path, [Query(query_id=1, text="q")]),
            index_result=index_result,
            top_k=1,
            aggregation="median",
        )


def test_custom_aggregation_must_return_finite_numeric_score(
    tmp_path,
    monkeypatch,
    chunks,
):
    collection = _collection(tmp_path, [Query(query_id=1, text="q")])
    encoder = FakeQueryEncoder([[1.0, 0.0]])
    index = FakeSearchIndex(
        dimension=2,
        ntotal=3,
        scores=[[0.9, 0.8, 0.3]],
        positions=[[0, 1, 2]],
    )
    index_result = _index_result(tmp_path, chunks, index)
    _patch_encoder(monkeypatch, encoder)

    with pytest.raises(RetrievalConfigurationError, match="numeric page score"):
        retrieve(
            collection=collection,
            index_result=index_result,
            aggregation=lambda chunk_results: "bad",
        )

    with pytest.raises(RetrievalConfigurationError, match="non-finite"):
        retrieve(
            collection=collection,
            index_result=index_result,
            aggregation=lambda chunk_results: float("nan"),
        )
