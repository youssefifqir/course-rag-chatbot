"""Load FAISS + embeddings and do top-k retrieval."""
from __future__ import annotations

import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, TOP_K, VECTORSTORE_DIR


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _ready() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(VECTORSTORE_DIR / "faiss.index"))
    with (VECTORSTORE_DIR / "metadata.jsonl").open(encoding="utf-8") as fh:
        metadata = [json.loads(line) for line in fh]
    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"index ({index.ntotal}) and metadata ({len(metadata)} disagree - rebuild."
        )
    return index, metadata


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k chunks: {rank, chunk_id, file, page, score, text}."""
    index, metadata = _ready()
    query_vec = _model().encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.ascontiguousarray(query_vec, dtype="float32"), top_k)

    hits = []
    for rank, (score, chunk_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if chunk_id == -1:  # faiss pad value
            continue
        meta = metadata[chunk_id]
        hits.append(
            {
                "rank": rank,
                "chunk_id": int(chunk_id),
                "file": meta["metadata"]["source"],
                "page": meta["metadata"]["page"],
                "score": float(score),
                "text": meta["text"],
            }
        )
    return hits
