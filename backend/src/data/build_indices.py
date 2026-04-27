"""One-time index build for ICD-10-CM and CPT corpora.

Run with:
  python -m src.data.build_indices

This creates:
  .chroma/                          - Chroma persistent client directory
  .chroma/icd_10_cm_docs.json       - Documents for BM25 (ICD-10)
  .chroma/cpt_docs.json             - Documents for BM25 (CPT)

Idempotent: re-running rebuilds the indices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = Path(__file__).resolve().parents[2] / ".chroma"


def _embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )


def _build_one(code_system: str, docs: list[dict[str, str]]) -> None:
    print(f"[{code_system}] Building indices for {len(docs):,} codes...")

    # 1) Persist docs JSON for BM25 to load later (deterministic, no embeddings).
    CHROMA_DIR.mkdir(exist_ok=True)
    docs_path = CHROMA_DIR / f"{code_system}_docs.json"
    with docs_path.open("w") as f:
        json.dump(docs, f)

    # 2) Build Chroma collection. Drop and recreate to avoid stale embeddings.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = f"{code_system.lower().replace('-', '_')}_codes"
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=_embedding_fn(),
    )

    # 3) Add in batches; large corpora otherwise OOM the embedding model.
    BATCH = 256
    for i in range(0, len(docs), BATCH):
        chunk = docs[i : i + BATCH]
        collection.add(
            ids=[d["code"] for d in chunk],
            documents=[d["description"] for d in chunk],
        )
        if (i // BATCH) % 20 == 0:
            print(f"  embedded {min(i + BATCH, len(docs)):,} / {len(docs):,}")

    print(f"[{code_system}] Done.")


def main() -> int:
    from .load_cpt_demo import load_cpt_codes
    from .load_icd10 import load_icd10_codes

    try:
        icd10 = load_icd10_codes()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    cpt = load_cpt_codes()

    _build_one("ICD-10-CM", icd10)
    _build_one("CPT", cpt)

    print("\nAll indices built. The pipeline is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
