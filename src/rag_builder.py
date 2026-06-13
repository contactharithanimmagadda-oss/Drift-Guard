"""
src/rag_builder.py
==================
Person B — Day 2 (class skeleton + ingest) | Day 3 (build, save, query, load)

Three specialised FAISS RAG pipelines for DriftGuard:

  Pipeline A — Lineage       IndexIVFFlat   BAAI/bge-base-en-v1.5      256-token chunks
  Pipeline B — Knowledge     IndexHNSWFlat  all-MiniLM-L6-v2           512-token chunks
  Pipeline C — Model Meta    IndexFlatL2    msmarco-distilbert-base-v4 128-token chunks

Each pipeline has:
  - ingest_folder()  read .txt/.md/.json files, chunk them, store in self.chunks
  - build()          encode all chunks -> FAISS index
  - save()           write index.faiss + chunks.pkl to disk
  - load() (static)  reload from disk without re-encoding
  - query()          encode a query string -> return top-k chunks

Module-level helper:
  - load_pipelines() -> (rag_a, rag_b, rag_c)  called by Person A in tools.py

Colab notes:
  - batch_size=32 in encode() — avoids OOM on 12 GB free-tier RAM
  - After build(), always save to Drive:
      !cp -r ./indexes /content/drive/MyDrive/steelguard_prod/driftguard/
  - On session restart, call load_pipelines() — no re-encoding needed (<5s)
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path

# Lazy imports — allows mock injection in tests before torch loads
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _ST_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _LC_AVAILABLE = True
except Exception:
    RecursiveCharacterTextSplitter = None
    _LC_AVAILABLE = False


# ── PIPELINE CONFIGS (module-level constants) ─────────────────────────────────

PIPELINE_CONFIGS = [
    {
        "name":          "lineage",
        "model_name":    "BAAI/bge-base-en-v1.5",
        "index_type":    "IVFFlat",
        "index_params":  {"nlist": 64, "nprobe": 8},
        "chunk_size":    256,
        "overlap":       25,
    },
    {
        "name":          "knowledge",
        "model_name":    "sentence-transformers/all-MiniLM-L6-v2",
        "index_type":    "HNSWFlat",
        "index_params":  {"M": 16, "efC": 100, "efS": 32},
        "chunk_size":    512,
        "overlap":       77,
    },
    {
        "name":          "model_meta",
        "model_name":    "sentence-transformers/msmarco-distilbert-base-v4",
        "index_type":    "FlatL2",
        "index_params":  {},
        "chunk_size":    128,
        "overlap":       0,
    },
]

# Map pipeline name -> knowledge folder (relative to base_dir passed to build_all_pipelines)
FOLDER_MAP = {
    "lineage":    "knowledge/lineage",
    "knowledge":  "knowledge/runbooks",
    "model_meta": "knowledge/model_meta",
}

# Map pipeline name -> indexes/ subfolder
INDEX_MAP = {
    "lineage":    "indexes/pipeline_A",
    "knowledge":  "indexes/pipeline_B",
    "model_meta": "indexes/pipeline_C",
}


# ── RAG PIPELINE CLASS ────────────────────────────────────────────────────────

class RAGPipeline:
    """
    A single FAISS-backed RAG pipeline.

    Typical usage (build once, reload forever):
        # BUILD (Person B, Day 3)
        pipe = RAGPipeline(**cfg)
        pipe.ingest_folder("data/knowledge/lineage")
        pipe.build()
        pipe.save("indexes/pipeline_A")

        # RELOAD (Person A, Day 5 — no re-encoding)
        pipe = RAGPipeline.load("indexes/pipeline_A", **cfg)
        results = pipe.query("sensor recalibration 2024", k=3)
    """

    def __init__(self, name: str, model_name: str, index_type: str,
                 index_params: dict, chunk_size: int, overlap: int):
        self.name        = name
        self.model_name  = model_name
        self.index_type  = index_type
        self.params      = index_params
        self.chunk_size  = chunk_size
        self.overlap     = overlap
        self.chunks      = []   # list of {"text": str, "source": str}
        self.index       = None
        # Load embedding model (downloads on first use, cached after)
        print(f"[rag_builder] Loading embedding model: {model_name}")
        _ST = SentenceTransformer if SentenceTransformer is not None else getattr(__builtins__, '_mock_st', None)
        if _ST is None:
            raise ImportError("sentence-transformers not available. Install it with: pip install sentence-transformers")
        self.model = _ST(model_name)
        self.dim   = self.model.get_embedding_dimension()
        print(f"[rag_builder] Pipeline '{name}' ready | dim={self.dim}")

    # ── Ingest ──────────────────────────────────────────────────────────────

    def ingest_folder(self, folder_path: str) -> int:
        """
        Read all .txt / .md / .json files in folder_path.
        Split into chunks and store in self.chunks.

        Returns number of chunks created.
        """
        import re as _re
        if RecursiveCharacterTextSplitter is not None:
            _sp = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.overlap,
                length_function=len, separators=["\n\n", "\n", ". ", " ", ""],
            )
            _split_fn = _sp.split_text
        else:
            def _split_fn(text):
                parts, out = _re.split(r"\n\n+", text), []
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    step = max(1, self.chunk_size - self.overlap)
                    for i in range(0, len(p), step):
                        out.append(p[i:i+self.chunk_size])
                return out
        n_before = len(self.chunks)
        folder   = Path(folder_path)

        if not folder.exists():
            print(f"[rag_builder] WARNING: folder not found: {folder_path}")
            return 0

        for fpath in sorted(folder.glob("**/*")):
            if fpath.suffix not in {".txt", ".md", ".json"}:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"[rag_builder] Could not read {fpath}: {e}")
                continue

            if not text:
                continue

            for chunk in _split_fn(text):
                # Skip chunks that are too short to be useful
                if len(chunk.split()) >= 8:
                    self.chunks.append({
                        "text":   chunk,
                        "source": str(fpath),
                    })

        n_added = len(self.chunks) - n_before
        print(f"[rag_builder] Ingested {folder.name}/ -> {n_added} chunks "
              f"(total: {len(self.chunks)})")
        return n_added

    # ── Build ────────────────────────────────────────────────────────────────

    def _make_index(self) -> faiss.Index:
        """Create the FAISS index based on index_type and params."""
        p = self.params
        if self.index_type == "FlatL2":
            return faiss.IndexFlatL2(self.dim)

        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dim)
            idx       = faiss.IndexIVFFlat(quantizer, self.dim, p["nlist"])
            idx.nprobe = p["nprobe"]
            return idx

        elif self.index_type == "HNSWFlat":
            idx = faiss.IndexHNSWFlat(self.dim, p["M"])
            idx.hnsw.efConstruction = p["efC"]
            idx.hnsw.efSearch       = p["efS"]
            return idx

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

    def build(self) -> None:
        """
        Encode all ingested chunks and build the FAISS index.

        Colab note: batch_size=32 avoids OOM on free-tier 12 GB RAM.
        On Pro tier you can safely use batch_size=64.
        """
        if not self.chunks:
            raise RuntimeError(f"[rag_builder] No chunks to build. Run ingest_folder() first.")

        texts = [c["text"] for c in self.chunks]
        print(f"[rag_builder] Encoding {len(texts)} chunks with {self.model_name}...")

        vecs = self.model.encode(
            texts,
            batch_size=32,           # safe for Colab free tier
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vecs = np.array(vecs, dtype=np.float32)

        # IVFFlat requires nlist <= n_vectors — scale down for small corpora
        if self.index_type == "IVFFlat":
            max_nlist = max(4, len(vecs) // 4)
            if self.params.get("nlist", 64) > max_nlist:
                original = self.params["nlist"]
                self.params = {**self.params, "nlist": max_nlist}
                print(f"[rag_builder] Auto-scaled nlist {original} -> {max_nlist} "
                      f"(corpus has only {len(vecs)} vectors)")

        self.index = self._make_index()

        # IVFFlat requires training before adding vectors
        if hasattr(self.index, "train"):
            print(f"[rag_builder] Training {self.index_type} index...")
            self.index.train(vecs)

        self.index.add(vecs)
        print(f"[rag_builder] Built '{self.name}' index | "
              f"type={self.index_type} | ntotal={self.index.ntotal}")

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save index.faiss + chunks.pkl + meta.json to disk.
        After saving on Colab, always copy to Drive:
            !cp -r ./indexes /content/drive/MyDrive/steelguard_prod/driftguard/
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(p / "index.faiss"))

        with open(p / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        meta = {
            "name":        self.name,
            "model_name":  self.model_name,
            "index_type":  self.index_type,
            "index_params":self.params,
            "chunk_size":  self.chunk_size,
            "overlap":     self.overlap,
            "n_chunks":    len(self.chunks),
            "dim":         self.dim,
        }
        with open(p / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[rag_builder] Saved '{self.name}' -> {path}/ "
              f"({len(self.chunks)} chunks, {self.index.ntotal} vectors)")

    @staticmethod
    def load(path: str, model_name: str, index_type: str,
             index_params: dict, chunk_size: int, overlap: int,
             name: str = "") -> "RAGPipeline":
        """
        Reload a saved pipeline from disk.
        No re-encoding — loads in under 5 seconds even on Colab free tier.

        IMPORTANT: nprobe and efSearch are not stored in the FAISS binary.
        This method restores them from index_params automatically.

        Usage:
            rag_a = RAGPipeline.load(
                path="indexes/pipeline_A",
                model_name="BAAI/bge-base-en-v1.5",
                index_type="IVFFlat",
                index_params={"nlist": 64, "nprobe": 8},
                chunk_size=256, overlap=25,
            )
        """
        p = Path(path)
        if not (p / "index.faiss").exists():
            raise FileNotFoundError(
                f"No index.faiss found at {path}. "
                "Run build() + save() first, or copy indexes from Drive."
            )

        # Use __new__ to skip __init__'s model loading
        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe.name        = name or p.name
        pipe.model_name  = model_name
        pipe.index_type  = index_type
        pipe.params      = index_params
        pipe.chunk_size  = chunk_size
        pipe.overlap     = overlap

        print(f"[rag_builder] Loading model for reload: {model_name}")
        pipe.model = SentenceTransformer(model_name)
        pipe.dim   = pipe.model.get_embedding_dimension()

        pipe.index = faiss.read_index(str(p / "index.faiss"))

        # Restore runtime params not saved in FAISS binary
        if index_type == "IVFFlat" and "nprobe" in index_params:
            pipe.index.nprobe = index_params["nprobe"]
        if index_type == "HNSWFlat" and "efS" in index_params:
            pipe.index.hnsw.efSearch = index_params["efS"]

        with open(p / "chunks.pkl", "rb") as f:
            pipe.chunks = pickle.load(f)

        print(f"[rag_builder] Loaded '{pipe.name}' | "
              f"ntotal={pipe.index.ntotal} | chunks={len(pipe.chunks)}")
        return pipe

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, query_text: str, k: int = 5) -> list:
        """
        Encode query_text and search the FAISS index.

        Returns list of dicts:
            [{"score": float, "text": str, "source": str}, ...]
        Sorted by score ascending (lower = more similar for L2/cosine).
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Run build() or load() first.")

        q_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        q_vec = np.array(q_vec, dtype=np.float32)

        D, I = self.index.search(q_vec, k)

        results = []
        for rank in range(k):
            idx = I[0][rank]
            if idx < 0:     # FAISS returns -1 for unfilled slots
                continue
            results.append({
                "score":  round(float(D[0][rank]), 6),
                "text":   self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
            })
        return results


# ── MODULE-LEVEL HELPER ───────────────────────────────────────────────────────

def load_pipelines(base_path: str = "indexes") -> tuple:
    """
    Load all 3 saved FAISS pipelines from disk.

    Returns: (rag_a, rag_b, rag_c)
        rag_a  Pipeline A — Lineage   (IndexIVFFlat,  bge-base-en-v1.5)
        rag_b  Pipeline B — Knowledge (IndexHNSWFlat, all-MiniLM-L6-v2)
        rag_c  Pipeline C — Model meta(IndexFlatL2,   msmarco-distilbert)

    Called by Person A in tools.py on Day 5 to replace mock RAG calls:
        rag_a, rag_b, rag_c = load_pipelines()

    Raises FileNotFoundError if indexes have not been built yet.
    """
    rag_a = RAGPipeline.load(
        path=f"{base_path}/pipeline_A",
        name="lineage",
        model_name="BAAI/bge-base-en-v1.5",
        index_type="IVFFlat",
        index_params={"nlist": 64, "nprobe": 8},
        chunk_size=256,
        overlap=25,
    )
    rag_b = RAGPipeline.load(
        path=f"{base_path}/pipeline_B",
        name="knowledge",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="HNSWFlat",
        index_params={"M": 16, "efC": 100, "efS": 32},
        chunk_size=512,
        overlap=77,
    )
    rag_c = RAGPipeline.load(
        path=f"{base_path}/pipeline_C",
        name="model_meta",
        model_name="sentence-transformers/msmarco-distilbert-base-v4",
        index_type="FlatL2",
        index_params={},
        chunk_size=128,
        overlap=0,
    )
    print(f"\n[rag_builder] All 3 pipelines loaded: "
          f"A={rag_a.index.ntotal}v  B={rag_b.index.ntotal}v  C={rag_c.index.ntotal}v")
    return rag_a, rag_b, rag_c


def build_all_pipelines(base_dir: str = ".") -> tuple:
    """
    Build all 3 pipelines from scratch and save them.

    Called by Person B in person_b_complete.ipynb Cell 4.
    Run once; reload with load_pipelines() afterward.

    Args:
        base_dir: root of the driftguard/ folder

    Returns: (rag_a, rag_b, rag_c)
    """
    pipelines = []
    for cfg in PIPELINE_CONFIGS:
        pipe = RAGPipeline(**cfg)
        folder = str(Path(base_dir) / FOLDER_MAP[cfg["name"]])
        pipe.ingest_folder(folder)
        pipe.build()
        save_path = str(Path(base_dir) / INDEX_MAP[cfg["name"]])
        pipe.save(save_path)
        pipelines.append(pipe)

    rag_a, rag_b, rag_c = pipelines
    print(f"\n[rag_builder] Build complete:")
    print(f"  Pipeline A (lineage):    {rag_a.index.ntotal} vectors  -> {INDEX_MAP['lineage']}")
    print(f"  Pipeline B (knowledge):  {rag_b.index.ntotal} vectors  -> {INDEX_MAP['knowledge']}")
    print(f"  Pipeline C (model_meta): {rag_c.index.ntotal} vectors  -> {INDEX_MAP['model_meta']}")
    return rag_a, rag_b, rag_c
