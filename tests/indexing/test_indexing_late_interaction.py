"""Focused tests for Sentence Transformers late-interaction indexing."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
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


class FakeMultiVectorEncoder:
    def __init__(self):
        self.calls = []

    def encode_document(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [
            np.ones((2, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        ][: len(texts)]


class FakeDenseEncoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeFastPlaid:
    def __init__(self):
        self.embeddings = None
        self.create_calls = 0

    def create(self, *, documents_embeddings):
        self.create_calls += 1
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
    encoder = FakeMultiVectorEncoder()
    fast_plaid = FakeFastPlaid()

    import ragit.indexing.late_interaction as module

    monkeypatch.setattr(
        module,
        "_create_fast_plaid_index",
        lambda **kwargs: fast_plaid,
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
    assert fast_plaid.create_calls == 1
    assert [embedding.shape for embedding in fast_plaid.embeddings] == [(2, 4), (3, 4)]

    texts, kwargs = encoder.calls[0]
    assert texts == ["alpha", "beta"]
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is True

    assert (result.output_path / "config.json").exists()
    assert (result.output_path / "chunks.jsonl").exists()
    assert (result.output_path / "index.ready").exists()


def test_late_interaction_build_recomputes_existing_index(
    tmp_path,
    monkeypatch,
):
    import ragit.indexing.late_interaction as module

    created = []

    def fake_create(**kwargs):
        created.append(kwargs)
        return FakeFastPlaid()

    monkeypatch.setattr(module, "_create_fast_plaid_index", fake_create)

    config = IndexingConfig(
        index_type="late_interaction",
        model_name="fake/colbert",
    )
    collection = SimpleNamespace(pdfs_path=tmp_path)
    first_encoder = FakeMultiVectorEncoder()
    second_encoder = FakeMultiVectorEncoder()

    first = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path),
        config=config,
        encoder=first_encoder,
    )
    second = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path),
        config=config,
        encoder=second_encoder,
    )

    assert first.output_path == second.output_path
    assert len(first_encoder.calls) == 1
    assert len(second_encoder.calls) == 1
    assert created[-1]["output_path"] == second.output_path


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
        def encode_document(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    import ragit.indexing.late_interaction as module

    monkeypatch.setattr(
        module,
        "_create_fast_plaid_index",
        lambda **kwargs: FakeFastPlaid(),
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
            encoder=FakeMultiVectorEncoder(),
        )


def test_late_interaction_hides_fast_plaid_tuning_options(tmp_path):
    with pytest.raises(IndexingConfigurationError, match="Unsupported"):
        build_index(
            collection=SimpleNamespace(pdfs_path=tmp_path),
            chunking_result=_chunking_result(tmp_path),
            config=IndexingConfig(
                index_type="late_interaction",
                model_name="fake/colbert",
                options={"n_ivf_probe": 32},
            ),
            encoder=FakeMultiVectorEncoder(),
        )


def test_fast_plaid_artifacts_are_versioned_separately_from_legacy_pylate():
    from ragit.indexing.storage import indexing_configuration_hash

    config = IndexingConfig(
        index_type="late_interaction",
        model_name="lightonai/LateOn",
    )
    legacy_payload = {
        "indexing_config": config.model_dump(mode="json"),
        "chunking_config_hash": "chunking-hash",
    }

    legacy_hash = hashlib.sha256(
        json.dumps(
            legacy_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert indexing_configuration_hash(config, "chunking-hash") != legacy_hash


def test_late_interaction_creates_fast_plaid_index(monkeypatch, tmp_path):
    import ragit.indexing.late_interaction as module

    captured = {}

    class FastPlaid:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fast_plaid = ModuleType("fast_plaid")
    fast_plaid.search = SimpleNamespace(FastPlaid=FastPlaid)
    monkeypatch.setitem(sys.modules, "fast_plaid", fast_plaid)

    index = module._create_fast_plaid_index(
        output_path=tmp_path / "index",
        config=IndexingConfig(
            index_type="late_interaction",
            model_name="lightonai/LateOn",
            options={"device": "cuda"},
        ),
    )

    assert isinstance(index, FastPlaid)
    assert captured == {"index": str(tmp_path / "index"), "device": "cuda"}


def test_late_interaction_uses_sentence_transformers_multivector_encoder(
    monkeypatch,
):
    import ragit.indexing.late_interaction as module

    captured = {}

    class MultiVectorEncoder:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.MultiVectorEncoder = MultiVectorEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    encoder = module._load_encoder(
        IndexingConfig(
            index_type="late_interaction",
            model_name="lightonai/LateOn",
            options={"device": "cuda", "trust_remote_code": True},
        )
    )

    assert isinstance(encoder, MultiVectorEncoder)
    assert captured == {
        "model_name": "lightonai/LateOn",
        "kwargs": {"device": "cuda", "trust_remote_code": True},
    }


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
