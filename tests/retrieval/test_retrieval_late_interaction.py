"""Focused tests for late-interaction page-level retrieval."""

import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np

from ragit.chunking.models import Chunk
from ragit.data.models import Query
from ragit.indexing import IndexingConfig, LateInteractionIndexResult
from ragit.retrieval import retrieve


class FakeMultiVectorEncoder:
    def __init__(self):
        self.calls = []

    def encode_query(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [
            np.ones((2, 4), dtype=np.float32)
            for _ in texts
        ]


class FakeFastPlaid:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


def _chunks():
    return [
        Chunk(
            chunk_id="doc:00000000",
            doc_id="doc",
            page_ids=[10],
            content="alpha",
        ),
        Chunk(
            chunk_id="doc:00000001",
            doc_id="doc",
            page_ids=[10, 11],
            content="beta",
        ),
        Chunk(
            chunk_id="doc:00000002",
            doc_id="doc",
            page_ids=[12],
            content="gamma",
        ),
    ]


def _index_result(tmp_path: Path, index=None) -> LateInteractionIndexResult:
    chunks = _chunks()
    return LateInteractionIndexResult(
        index=object() if index is None else index,
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        config_hash="li-index-hash",
        output_path=tmp_path / ".ragit" / "indexing" / "late_interaction" / "x",
        config=IndexingConfig(
            index_type="late_interaction",
            model_name="fake/colbert",
            options={"batch_size": 8},
        ),
        chunks=chunks,
    )


def test_late_interaction_retrieval_uses_query_encoding_and_page_aggregation(
    tmp_path,
    monkeypatch,
):
    encoder = FakeMultiVectorEncoder()
    fast_plaid = FakeFastPlaid(
        [[
            (1, 0.9),
            (0, 0.8),
            (2, 0.2),
        ]]
    )

    import ragit.retrieval.dense as module

    monkeypatch.setattr(
        module,
        "_load_late_interaction_encoder",
        lambda result: encoder,
    )

    collection = SimpleNamespace(
        pdfs_path=tmp_path,
        queries=[Query(query_id=7, text="question")],
    )

    result = retrieve(
        collection=collection,
        index_result=_index_result(tmp_path, index=fast_plaid),
        top_k=2,
        aggregation="max",
    )

    texts, kwargs = encoder.calls[0]
    assert texts == ["question"]
    assert kwargs["batch_size"] == 8

    call = fast_plaid.calls[0]
    assert call["top_k"] == 3
    assert call["show_progress"] is False

    assert [(page.page_id, page.rank) for page in result.results] == [
        (10, 1),
        (11, 2),
    ]
    assert [page.score for page in result.results] == [0.9, 0.9]

    page_10 = result.results[0]
    assert [chunk.chunk_id for chunk in page_10.chunks] == [
        "doc:00000001",
        "doc:00000000",
    ]
    assert [chunk.rank for chunk in page_10.chunks] == [1, 2]
    assert result.output_path.parent.name == "late_interaction"


def test_late_interaction_sum_uses_all_contributing_chunks(
    tmp_path,
    monkeypatch,
):
    encoder = FakeMultiVectorEncoder()
    fast_plaid = FakeFastPlaid(
        [[
            (0, 0.8),
            (1, 0.7),
            (2, 0.1),
        ]]
    )

    import ragit.retrieval.dense as module

    monkeypatch.setattr(
        module,
        "_load_late_interaction_encoder",
        lambda result: encoder,
    )

    result = retrieve(
        collection=SimpleNamespace(
            pdfs_path=tmp_path,
            queries=[Query(query_id=1, text="q")],
        ),
        index_result=_index_result(tmp_path, index=fast_plaid),
        top_k=10,
        aggregation="sum",
    )

    scores = {page.page_id: page.score for page in result.results}
    assert scores[10] == 1.5
    assert scores[11] == 0.7
    assert scores[12] == 0.1


def test_late_interaction_persists_same_page_level_schema(
    tmp_path,
    monkeypatch,
):
    encoder = FakeMultiVectorEncoder()
    fast_plaid = FakeFastPlaid(
        [[
            (0, 0.8),
            (1, 0.7),
            (2, 0.1),
        ]]
    )

    import json
    import ragit.retrieval.dense as module

    monkeypatch.setattr(
        module,
        "_load_late_interaction_encoder",
        lambda result: encoder,
    )

    result = retrieve(
        collection=SimpleNamespace(
            pdfs_path=tmp_path,
            queries=[Query(query_id=1, text="q")],
        ),
        index_result=_index_result(tmp_path, index=fast_plaid),
        top_k=2,
    )

    payload = json.loads(
        (result.output_path / "results.json").read_text(encoding="utf-8")
    )
    assert payload["queries"][0]["query_id"] == 1
    assert payload["queries"][0]["pages"][0]["page_id"] == 10
    assert payload["queries"][0]["pages"][0]["chunks"][0]["chunk_id"]



def test_late_interaction_uses_internal_candidate_depth_and_accepts_partial_ranking(
    tmp_path,
    monkeypatch,
):
    chunks = [
        Chunk(
            chunk_id=f"doc:{index:08d}",
            doc_id="doc",
            page_ids=[index],
            content=f"chunk {index}",
        )
        for index in range(150)
    ]
    index_result = LateInteractionIndexResult(
        index=object(),
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        config_hash="li-large-index-hash",
        output_path=(
            tmp_path / ".ragit" / "indexing" / "late_interaction" / "large"
        ),
        config=IndexingConfig(
            index_type="late_interaction",
            model_name="fake/colbert",
            options={"batch_size": 8},
        ),
        chunks=chunks,
    )

    encoder = FakeMultiVectorEncoder()
    fast_plaid = FakeFastPlaid(
        [[
            (5, 0.9),
            (12, 0.8),
        ]]
    )

    import ragit.retrieval.dense as module

    monkeypatch.setattr(
        module,
        "_load_late_interaction_encoder",
        lambda result: encoder,
    )
    index_result.index = fast_plaid

    result = retrieve(
        collection=SimpleNamespace(
            pdfs_path=tmp_path,
            queries=[Query(query_id=1, text="q")],
        ),
        index_result=index_result,
        top_k=10,
        aggregation="max",
    )

    call = fast_plaid.calls[0]
    assert call["top_k"] == 150
    assert call["show_progress"] is False

    assert [(page.page_id, page.rank) for page in result.results] == [
        (5, 1),
        (12, 2),
    ]
    assert [page.chunks[0].rank for page in result.results] == [1, 2]


def test_late_interaction_retrieval_loads_sentence_transformers_encoder(
    tmp_path,
    monkeypatch,
):
    import ragit.retrieval.dense as module

    captured = {}

    class MultiVectorEncoder:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.MultiVectorEncoder = MultiVectorEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    encoder = module._load_late_interaction_encoder(_index_result(tmp_path))

    assert isinstance(encoder, MultiVectorEncoder)
    assert captured == {
        "model_name": "fake/colbert",
        "kwargs": {},
    }
