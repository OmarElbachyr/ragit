"""Focused tests for PyLate late-interaction indexing."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ragit.chunking.models import Chunk, ChunkingResult
from ragit.indexing import (
    IndexingConfig,
    IndexingConfigurationError,
    LateInteractionIndexResult,
    build_index,
)


class FakeColBERT:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [
            np.ones((2, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        ][: len(texts)]


class FakeDenseEncoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


class FakePLAID:
    def __init__(self):
        self.ids = None
        self.embeddings = None
        self.add_calls = 0

    def add_documents(self, *, documents_ids, documents_embeddings):
        self.add_calls += 1
        self.ids = list(documents_ids)
        self.embeddings = list(documents_embeddings)


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
            page_ids=[11, 12],
            content="beta",
        ),
    ]


def _chunking_result(tmp_path: Path) -> ChunkingResult:
    return ChunkingResult(
        chunks=_chunks(),
        config_hash="chunking-hash",
        parsing_config_hash="parsing-hash",
        output_path=tmp_path / "chunks",
    )


def test_late_interaction_index_encodes_multivectors_and_preserves_chunk_ids(
    tmp_path,
    monkeypatch,
):
    encoder = FakeColBERT()
    plaid = FakePLAID()

    import ragit.indexing.late_interaction as module

    monkeypatch.setattr(
        module,
        "_create_pylate_index",
        lambda **kwargs: plaid,
    )

    config = IndexingConfig(
        index_type="late_interaction",
        model_name="fake/colbert",
        options={"batch_size": 8, "show_progress_bar": True},
    )
    collection = SimpleNamespace(pdfs_path=tmp_path)

    result = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path),
        config=config,
        encoder=encoder,
    )

    assert isinstance(result, LateInteractionIndexResult)
    assert result.chunk_ids == ["doc:00000000", "doc:00000001"]
    assert plaid.ids == result.chunk_ids
    assert plaid.add_calls == 1
    assert [embedding.shape for embedding in plaid.embeddings] == [(2, 4), (3, 4)]

    texts, kwargs = encoder.calls[0]
    assert texts == ["alpha", "beta"]
    assert kwargs["is_query"] is False
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is True

    assert (result.output_path / "config.json").exists()
    assert (result.output_path / "chunks.jsonl").exists()
    assert (result.output_path / "index.ready").exists()


def test_dense_encoder_is_rejected_for_late_interaction(tmp_path):
    config = IndexingConfig(
        index_type="late_interaction",
        model_name="fake/dense",
    )
    collection = SimpleNamespace(pdfs_path=tmp_path)

    with pytest.raises(IndexingConfigurationError, match="late-interaction"):
        build_index(
            collection=collection,
            chunking_result=_chunking_result(tmp_path),
            config=config,
            encoder=FakeDenseEncoder(),
        )


def test_late_interaction_rejects_dense_shaped_encoder_output(
    tmp_path,
    monkeypatch,
):
    class BrokenColBERT:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    import ragit.indexing.late_interaction as module

    monkeypatch.setattr(
        module,
        "_create_pylate_index",
        lambda **kwargs: FakePLAID(),
    )

    config = IndexingConfig(
        index_type="late_interaction",
        model_name="fake/broken-colbert",
    )

    with pytest.raises(IndexingConfigurationError, match="2D token"):
        build_index(
            collection=SimpleNamespace(pdfs_path=tmp_path),
            chunking_result=_chunking_result(tmp_path),
            config=config,
            encoder=BrokenColBERT(),
        )


def test_late_interaction_options_are_validated(tmp_path):
    config = IndexingConfig(
        index_type="late_interaction",
        model_name="fake/colbert",
        options={"unknown": True},
    )

    with pytest.raises(IndexingConfigurationError, match="Unsupported"):
        build_index(
            collection=SimpleNamespace(pdfs_path=tmp_path),
            chunking_result=_chunking_result(tmp_path),
            config=config,
            encoder=FakeColBERT(),
        )


def test_build_index_dispatches_to_late_interaction(tmp_path, monkeypatch):
    import ragit.indexing.api as api

    expected = object()
    monkeypatch.setattr(
        api,
        "_build_late_interaction_index_from_result",
        lambda **kwargs: expected,
    )

    result = build_index(
        collection=SimpleNamespace(pdfs_path=tmp_path),
        chunking_result=_chunking_result(tmp_path),
        config=IndexingConfig(
            index_type="late_interaction",
            model_name="fake/colbert",
        ),
    )
    assert result is expected
