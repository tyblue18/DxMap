"""Hybrid retrieval over the ICD-10-CM and CPT code corpora.

We use two retrievers in parallel:

  1. BM25 (rank-bm25) for keyword/exact matches. This is critical because many
     ICD-10-CM descriptions contain rare medical terms that an embedding model
     may not have seen often (e.g., "Type 1 diabetes mellitus with diabetic
     chronic kidney disease, stage 4").

  2. Dense embedding retrieval (Chroma + BAAI/bge-small-en-v1.5) for semantic
     matches. This catches paraphrases ("high blood pressure" → "Essential
     hypertension") that BM25 misses.

Results are merged with Reciprocal Rank Fusion (RRF), which is a simple,
parameter-light fusion method that consistently outperforms either retriever
alone in published benchmarks on clinical and general-domain retrieval.

Why bge-small-en-v1.5: it's a strong open-source embedding model, small enough
to run on CPU, and Apache-2.0 licensed. For higher quality, swap in
medcpt-query/article from NCBI which is medical-domain-specific.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from .schema import CodeCandidate

_CHROMA_DIR = Path(__file__).resolve().parents[2] / ".chroma"
_BM25_CACHE: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _tokenize(s: str) -> list[str]:
    """Cheap tokenizer for BM25. Real systems use a proper clinical tokenizer."""
    return [t for t in s.lower().replace(",", " ").replace(".", " ").split() if t]


@lru_cache(maxsize=1)
def _embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )


def _chroma_collection(code_system: str):
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    return client.get_or_create_collection(
        name=f"{code_system.lower().replace('-', '_')}_codes",
        embedding_function=_embedding_fn(),
    )


def _load_bm25(code_system: str) -> tuple[BM25Okapi, list[dict]]:
    """Load (and cache) the BM25 index. Documents are stored as JSON next to the index."""
    if code_system in _BM25_CACHE:
        return _BM25_CACHE[code_system]

    docs_path = _CHROMA_DIR / f"{code_system}_docs.json"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {docs_path}. Run `python -m src.data.build_indices` first."
        )

    with docs_path.open() as f:
        docs = json.load(f)

    tokenized = [_tokenize(d["description"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    _BM25_CACHE[code_system] = (bm25, docs)
    return bm25, docs


def retrieve(
    note: str,
    code_system: str = "ICD-10-CM",
    top_k_bm25: int = 30,
    top_k_dense: int = 30,
    final_k: int = 20,
) -> list[CodeCandidate]:
    """Hybrid retrieval with reciprocal rank fusion.

    Returns up to ``final_k`` candidates merged from BM25 and dense retrieval.
    """
    # --- BM25 ---
    bm25, docs = _load_bm25(code_system)
    tokenized_query = _tokenize(note)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )[:top_k_bm25]
    bm25_results = [(docs[i]["code"], docs[i]["description"], score) for i, score in bm25_ranked]

    # --- Dense ---
    collection = _chroma_collection(code_system)
    dense_results_raw = collection.query(query_texts=[note], n_results=top_k_dense)
    dense_results = list(
        zip(
            dense_results_raw["ids"][0],
            dense_results_raw["documents"][0],
            dense_results_raw["distances"][0],
        )
    )
    # Convert distance to similarity-ish score; lower distance = more similar
    dense_results = [(code, desc, 1.0 / (1.0 + dist)) for code, desc, dist in dense_results]

    # --- Reciprocal Rank Fusion ---
    # RRF score for a doc = sum over retrievers of 1 / (k + rank). k=60 is the
    # standard constant from the original RRF paper (Cormack et al. 2009).
    K = 60
    rrf_scores: dict[str, float] = {}
    descriptions: dict[str, str] = {}

    for rank, (code, desc, _) in enumerate(bm25_results):
        rrf_scores[code] = rrf_scores.get(code, 0.0) + 1.0 / (K + rank)
        descriptions[code] = desc
    for rank, (code, desc, _) in enumerate(dense_results):
        rrf_scores[code] = rrf_scores.get(code, 0.0) + 1.0 / (K + rank)
        descriptions[code] = desc

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    return [
        CodeCandidate(
            code=code,
            description=descriptions[code],
            code_system=code_system,  # type: ignore[arg-type]
            retrieval_score=score,
            retrieval_method="rrf",
        )
        for code, score in fused
    ]
