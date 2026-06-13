"""
tests/test_rag_builder.py
Run: pytest tests/test_rag_builder.py -v

Uses a lightweight mock embedder instead of sentence-transformers so tests
run quickly without downloading 400 MB models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import faiss
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE               = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_LINEAGE  = os.path.join(BASE, "examples/steelguard/knowledge/lineage")
KNOWLEDGE_RUNBOOKS = os.path.join(BASE, "examples/steelguard/knowledge/runbooks")
KNOWLEDGE_META     = os.path.join(BASE, "examples/steelguard/knowledge/model_meta")
INDEXES_DIR        = os.path.join(BASE, "indexes")


# ---- Mock embedder ---------------------------------------------------------

def make_mock_model(dim=64):
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim
    mock.get_embedding_dimension.return_value = dim

    def fake_encode(texts, batch_size=32, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True):
        vecs = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng  = np.random.RandomState(seed)
            v    = rng.randn(dim).astype(np.float32)
            if normalize_embeddings:
                v /= (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    mock.encode.side_effect = fake_encode
    return mock


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture
def lineage_pipe():
    from src.rag_builder import RAGPipeline
    with patch("src.rag_builder.SentenceTransformer", return_value=make_mock_model(64)):
        pipe = RAGPipeline(
            name="lineage_test",
            model_name="BAAI/bge-base-en-v1.5",
            index_type="IVFFlat",
            index_params={"nlist": 4, "nprobe": 2},
            chunk_size=256,
            overlap=25,
        )
    pipe.model = make_mock_model(64)
    pipe.dim   = 64
    return pipe


@pytest.fixture
def knowledge_pipe():
    from src.rag_builder import RAGPipeline
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.name       = "knowledge_test"
    pipe.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    pipe.index_type = "HNSWFlat"
    pipe.params     = {"M": 4, "efC": 20, "efS": 10}
    pipe.chunk_size = 512
    pipe.overlap    = 77
    pipe.chunks     = []
    pipe.index      = None
    pipe.model      = make_mock_model(64)
    pipe.dim        = 64
    return pipe


@pytest.fixture
def flat_pipe():
    from src.rag_builder import RAGPipeline
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.name       = "meta_test"
    pipe.model_name = "msmarco-distilbert-base-v4"
    pipe.index_type = "FlatL2"
    pipe.params     = {}
    pipe.chunk_size = 128
    pipe.overlap    = 0
    pipe.chunks     = []
    pipe.index      = None
    pipe.model      = make_mock_model(64)
    pipe.dim        = 64
    return pipe


# ---- ingest_folder tests --------------------------------------------------

def test_ingest_lineage_creates_chunks(lineage_pipe):
    n = lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    assert n > 0, "No chunks created from lineage folder"
    assert len(lineage_pipe.chunks) > 0


def test_ingest_knowledge_creates_chunks(knowledge_pipe):
    n = knowledge_pipe.ingest_folder(KNOWLEDGE_RUNBOOKS)
    assert n > 0, "No chunks from runbooks folder"


def test_ingest_each_chunk_has_text_and_source(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    for chunk in lineage_pipe.chunks:
        assert "text"   in chunk
        assert "source" in chunk
        assert len(chunk["text"].split()) >= 8


def test_ingest_nonexistent_folder_returns_zero(lineage_pipe):
    n = lineage_pipe.ingest_folder("/nonexistent/path/that/does/not/exist")
    assert n == 0


def test_ingest_minimum_word_filter(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    for chunk in lineage_pipe.chunks:
        assert len(chunk["text"].split()) >= 8, f"Short chunk leaked: '{chunk['text']}'"


# ---- build tests ----------------------------------------------------------

def test_build_ivfflat_not_empty(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    assert lineage_pipe.index is not None
    assert lineage_pipe.index.ntotal == len(lineage_pipe.chunks)


def test_build_hnswflat_not_empty(knowledge_pipe):
    knowledge_pipe.ingest_folder(KNOWLEDGE_RUNBOOKS)
    knowledge_pipe.build()
    assert knowledge_pipe.index.ntotal > 0


def test_build_flatl2_not_empty(flat_pipe):
    flat_pipe.ingest_folder(KNOWLEDGE_META)
    flat_pipe.build()
    assert flat_pipe.index.ntotal > 0


def test_build_fails_without_chunks(lineage_pipe):
    with pytest.raises(RuntimeError, match="No chunks"):
        lineage_pipe.build()


def test_build_nprobe_restored(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    assert lineage_pipe.index.nprobe == 2


# ---- save / load tests ----------------------------------------------------

def test_save_creates_files(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    with tempfile.TemporaryDirectory() as tmp:
        lineage_pipe.save(tmp)
        assert (Path(tmp) / "index.faiss").exists()
        assert (Path(tmp) / "chunks.pkl").exists()
        assert (Path(tmp) / "meta.json").exists()


def test_save_load_same_ntotal(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    original_ntotal = lineage_pipe.index.ntotal
    original_chunks = len(lineage_pipe.chunks)

    with tempfile.TemporaryDirectory() as tmp:
        lineage_pipe.save(tmp)
        from src.rag_builder import RAGPipeline
        reloaded = RAGPipeline.__new__(RAGPipeline)
        reloaded.name       = "reloaded"
        reloaded.model_name = "BAAI/bge-base-en-v1.5"
        reloaded.index_type = "IVFFlat"
        reloaded.params     = {"nlist": 4, "nprobe": 2}
        reloaded.chunk_size = 256
        reloaded.overlap    = 25
        reloaded.model      = make_mock_model(64)
        reloaded.dim        = 64
        reloaded.index = faiss.read_index(str(Path(tmp) / "index.faiss"))
        reloaded.index.nprobe = 2
        with open(Path(tmp) / "chunks.pkl", "rb") as f:
            reloaded.chunks = pickle.load(f)

        assert reloaded.index.ntotal == original_ntotal
        assert len(reloaded.chunks) == original_chunks


# ---- query tests ----------------------------------------------------------

def test_query_returns_k_results(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    results = lineage_pipe.query("sensor recalibration", k=3)
    assert len(results) == 3


def test_query_result_structure(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    results = lineage_pipe.query("maintenance event", k=2)
    for r in results:
        assert "score"  in r
        assert "text"   in r
        assert "source" in r
        assert isinstance(r["score"], float)
        assert isinstance(r["text"],  str)
        assert len(r["text"]) > 0


def test_query_k_capped_by_index_size(flat_pipe):
    flat_pipe.ingest_folder(KNOWLEDGE_META)
    flat_pipe.build()
    n = flat_pipe.index.ntotal
    results = flat_pipe.query("model performance", k=n + 10)
    assert len(results) <= n


def test_query_without_build_raises(lineage_pipe):
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    with pytest.raises(RuntimeError, match="not built"):
        lineage_pipe.query("test", k=3)


# ---- Recall tests ---------------------------------------------------------

def test_lineage_recall_sensor_recalibration(lineage_pipe):
    """
    With a hash-based mock embedder there is no semantic ranking, so we verify:
    1. The pipeline ingested chunks that contain the sensor recalibration event.
    2. query() returns valid result objects (structure + count).
    The real semantic recall is validated by integration tests with real embedders.
    """
    lineage_pipe.ingest_folder(KNOWLEDGE_LINEAGE)
    lineage_pipe.build()
    # Verify the lineage corpus contains the target event
    all_ingested = " ".join(c["text"] for c in lineage_pipe.chunks).lower()
    assert any(kw in all_ingested for kw in ["recalib", "keyence", "sensor"]), (
        "Lineage corpus does not contain sensor recalibration event -- check knowledge/lineage files"
    )
    # Verify query returns valid results (semantic ranking not testable with mock embedder)
    results = lineage_pipe.query("sensor recalibration Keyence 2024", k=5)
    assert len(results) > 0, "query() returned no results"
    assert all("text" in r and "score" in r for r in results)


def test_knowledge_recall_runbook_procedure(knowledge_pipe):
    knowledge_pipe.ingest_folder(KNOWLEDGE_RUNBOOKS)
    knowledge_pipe.build()
    results = knowledge_pipe.query("sensor recalibration procedure PSI threshold", k=5)
    all_text = " ".join(r["text"] for r in results).lower()
    assert any(keyword in all_text for keyword in ["psi", "recalib", "runbook", "procedure"]), (
        f"Knowledge pipeline failed to retrieve runbook procedure.\n"
        f"Top result: {results[0]['text'][:200] if results else 'NO RESULTS'}"
    )


# ---- load_pipelines smoke test --------------------------------------------

def test_load_pipelines_raises_if_not_built():
    from src.rag_builder import load_pipelines
    with pytest.raises(FileNotFoundError):
        load_pipelines("/nonexistent/path/to/indexes")


def test_build_all_pipelines_function():
    from src.rag_builder import build_all_pipelines, RAGPipeline
    import src.rag_builder as rb

    with tempfile.TemporaryDirectory() as tmp:
        orig_configs = [c.copy() for c in rb.PIPELINE_CONFIGS]
        orig_folder_map = dict(rb.FOLDER_MAP)
        orig_index_map  = dict(rb.INDEX_MAP)

        # Point build_all_pipelines at steelguard knowledge + temp index dir
        rb.FOLDER_MAP = {
            "lineage":    "knowledge/lineage",
            "knowledge":  "knowledge/runbooks",
            "model_meta": "knowledge/model_meta",
        }
        rb.INDEX_MAP = {
            "lineage":    os.path.join(tmp, "pipeline_A"),
            "knowledge":  os.path.join(tmp, "pipeline_B"),
            "model_meta": os.path.join(tmp, "pipeline_C"),
        }
        for c in rb.PIPELINE_CONFIGS:
            if "nlist" in c.get("index_params", {}):
                c["index_params"]["nlist"] = 4
                c["index_params"]["nprobe"] = 2

        original_st = rb.SentenceTransformer
        rb.SentenceTransformer = lambda name: make_mock_model(64)

        try:
            rag_a, rag_b, rag_c = build_all_pipelines(
                os.path.join(BASE, "examples/steelguard")
            )
            assert rag_a.index.ntotal > 0, "Pipeline A index empty"
            assert rag_b.index.ntotal > 0, "Pipeline B index empty"
            assert rag_c.index.ntotal > 0, "Pipeline C index empty"
        finally:
            rb.SentenceTransformer = original_st
            rb.FOLDER_MAP = orig_folder_map
            rb.INDEX_MAP  = orig_index_map
            for i, c in enumerate(rb.PIPELINE_CONFIGS):
                rb.PIPELINE_CONFIGS[i] = orig_configs[i]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
