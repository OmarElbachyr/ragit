"""Focused tests for the dense indexing baseline."""

from __future__ import annotations

import json
import pickle
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from ragit.chunking import Chunk, ChunkingResult
from ragit.indexing import (
    IndexingConfig,
    IndexingConfigurationError,
    IndexingStorageError,
    build_index,
    load_index,
)


class FakeDenseEncoder:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.encode_calls = 0
        self.last_inputs = None
        self.last_kwargs = None

    def get_embedding_dimension(self) -> int:
        return int(self.embeddings.shape[-1])

    def encode(self, inputs, **kwargs):
        self.encode_calls += 1
        self.last_inputs = list(inputs)
        self.last_kwargs = kwargs
        return self.embeddings.copy()


class FakeMultiVectorEncoder(FakeDenseEncoder):
    is_multi_vector = True


class ShapeOnlyMultiVectorEncoder:
    def __init__(self):
        self.encode_calls = 0

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, inputs, **kwargs):
        self.encode_calls += 1
        return np.zeros((len(inputs), 3, 4), dtype=np.float32)


class FakeIndexFlatIP:
    def __init__(self, dimension: int):
        self.d = dimension
        self.ntotal = 0
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.ndim == 2
        assert vectors.shape[1] == self.d
        self.vectors = vectors.copy()
        self.ntotal = len(vectors)


class FakeFaiss:
    IndexFlatIP = FakeIndexFlatIP

    @staticmethod
    def write_index(index, path):
        with open(path, "wb") as file:
            pickle.dump(index, file)

    @staticmethod
    def read_index(path):
        with open(path, "rb") as file:
            return pickle.load(file)


@pytest.fixture(autouse=True)
def fake_faiss(monkeypatch):
    """Avoid depending on a platform-specific FAISS build in unit tests."""
    import ragit.indexing.dense as dense_module
    import ragit.indexing.storage as storage_module

    fake = FakeFaiss()
    monkeypatch.setattr(dense_module, "_import_faiss", lambda: fake)
    monkeypatch.setattr(storage_module, "_import_faiss", lambda: fake)
    return fake


@pytest.fixture
def collection(tmp_path):
    return SimpleNamespace(pdfs_path=tmp_path)


def _chunking_result(tmp_path, chunks, config_hash="chunk-hash-a"):
    return ChunkingResult(
        chunks=chunks,
        config_hash=config_hash,
        parsing_config_hash="parse-hash-a",
        output_path=tmp_path / ".ragit" / "chunking" / "fake" / config_hash,
    )


@pytest.fixture
def chunks():
    return [
        Chunk(
            chunk_id="doc-a:0",
            doc_id="doc-a",
            page_ids=[10],
            content="alpha",
        ),
        Chunk(
            chunk_id="doc-a:1",
            doc_id="doc-a",
            page_ids=[10, 11],
            content="beta",
        ),
    ]


def test_dense_build_normalizes_float32_and_preserves_mapping(tmp_path, collection, chunks):
    encoder = FakeDenseEncoder(
        np.array(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
    )
    config = IndexingConfig(
        model_name="fake/dense",
        options={"batch_size": 8},
    )

    result = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path, chunks),
        config=config,
        encoder=encoder,
    )

    assert isinstance(result.index, FakeIndexFlatIP)
    assert result.index.ntotal == 2
    assert result.index.vectors.dtype == np.float32
    assert np.allclose(
        np.linalg.norm(result.index.vectors, axis=1),
        np.ones(2),
    )
    assert np.allclose(result.index.vectors[0], [0.6, 0.8, 0.0])
    assert [item.index_position for item in result.chunk_mapping] == [0, 1]
    assert [item.chunk_id for item in result.chunk_mapping] == [
        "doc-a:0",
        "doc-a:1",
    ]
    assert encoder.last_inputs == ["alpha", "beta"]
    assert encoder.last_kwargs["convert_to_numpy"] is True
    assert encoder.last_kwargs["normalize_embeddings"] is False


def test_explicit_multi_vector_capability_is_rejected_before_encoding(
    tmp_path,
    collection,
    chunks,
):
    encoder = FakeMultiVectorEncoder(
        np.zeros((2, 4), dtype=np.float32)
    )
    config = IndexingConfig(model_name="fake/multi-vector")

    with pytest.raises(
        IndexingConfigurationError,
        match="multi-vector representations",
    ):
        build_index(
            collection=collection,
            chunking_result=_chunking_result(tmp_path, chunks),
            config=config,
            encoder=encoder,
        )

    assert encoder.encode_calls == 0


def test_multi_vector_output_shape_is_rejected(tmp_path, collection, chunks):
    encoder = ShapeOnlyMultiVectorEncoder()
    config = IndexingConfig(model_name="fake/shape-multi-vector")

    with pytest.raises(
        IndexingConfigurationError,
        match="multi-vector or otherwise non-dense",
    ):
        build_index(
            collection=collection,
            chunking_result=_chunking_result(tmp_path, chunks),
            config=config,
            encoder=encoder,
        )

    assert encoder.encode_calls == 1


def test_dense_config_rejects_unknown_options_before_encoding(tmp_path, collection, chunks):
    encoder = FakeDenseEncoder(
        np.ones((2, 3), dtype=np.float32)
    )
    config = IndexingConfig(
        model_name="fake/dense",
        options={"metric": "cosine"},
    )

    with pytest.raises(
        IndexingConfigurationError,
        match="Unsupported dense indexing options",
    ):
        build_index(
            collection=collection,
            chunking_result=_chunking_result(tmp_path, chunks),
            config=config,
            encoder=encoder,
        )

    assert encoder.encode_calls == 0


def test_dense_loader_supports_sentence_transformers_remote_code(
    monkeypatch,
):
    import ragit.indexing.dense as module

    captured = {}

    class SentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    encoder = module._load_encoder(
        IndexingConfig(
            model_name="perplexity-ai/pplx-embed-v1-0.6b",
            options={"device": "cuda", "trust_remote_code": True},
        )
    )

    assert isinstance(encoder, SentenceTransformer)
    assert captured == {
        "model_name": "perplexity-ai/pplx-embed-v1-0.6b",
        "kwargs": {"device": "cuda", "trust_remote_code": True},
    }


def test_persistence_round_trip_and_rebuild(tmp_path, collection, chunks):
    encoder = FakeDenseEncoder(
        np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    config = IndexingConfig(model_name="fake/dense")

    built = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path, chunks),
        config=config,
        encoder=encoder,
    )

    assert (built.output_path / "config.json").exists()
    assert (built.output_path / "index.faiss").exists()
    assert (built.output_path / "chunks.jsonl").exists()

    loaded = load_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path, chunks),
        config=config,
    )
    assert loaded.config_hash == built.config_hash
    assert [item.chunk_id for item in loaded.chunk_mapping] == [
        "doc-a:0",
        "doc-a:1",
    ]
    assert np.allclose(loaded.index.vectors, built.index.vectors)

    # Building again recomputes and replaces the artifact.
    second_encoder = FakeDenseEncoder(
        np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    )
    rebuilt = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path, chunks),
        config=config,
        encoder=second_encoder,
    )
    assert rebuilt.output_path == built.output_path
    assert second_encoder.encode_calls == 1
    assert np.allclose(rebuilt.index.vectors, [[0.0, 1.0], [1.0, 0.0]])


def test_index_hash_depends_on_originating_chunking_artifact():
    from ragit.indexing.storage import indexing_configuration_hash

    config = IndexingConfig(model_name="fake/dense")

    first = indexing_configuration_hash(config, "chunk-hash-a")
    second = indexing_configuration_hash(config, "chunk-hash-b")

    assert first != second


def test_build_index_does_not_accept_overwrite(tmp_path, collection, chunks):
    encoder = FakeDenseEncoder(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    config = IndexingConfig(model_name="fake/dense")

    with pytest.raises(TypeError, match="overwrite"):
        build_index(
            collection=collection,
            chunking_result=_chunking_result(tmp_path, chunks),
            config=config,
            encoder=encoder,
            overwrite=True,
        )


def test_saved_config_mismatch_is_rejected(tmp_path, collection, chunks):
    encoder = FakeDenseEncoder(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    config = IndexingConfig(model_name="fake/dense")

    result = build_index(
        collection=collection,
        chunking_result=_chunking_result(tmp_path, chunks),
        config=config,
        encoder=encoder,
    )

    config_path = result.output_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["chunking_config_hash"] = "tampered"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        IndexingStorageError,
        match="does not match",
    ):
        load_index(
            collection=collection,
            chunking_result=_chunking_result(tmp_path, chunks),
            config=config,
        )
