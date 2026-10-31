"""Load per-domain FAISS + embeddings and do top-k retrieval.

Domains are separate indexes (course notes vs PubMed/MedlinePlus, ...), so a
question never accidentally crosses domain boundaries."""
from __future__ import annotations

import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, TOP_K, VECTORSTORE_DIR

DEFAULT_DOMAIN = "course"


def available_domains() -> list[str]:
    if not VECTORSTORE_DIR.is_dir():
        return []
    return sorted(
        d.name for d in VECTORSTORE_DIR.iterdir()
        if d.is_dir() and (d / "faiss.index").exists()
    )


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=8)
def _ready(domain: str = DEFAULT_DOMAIN) -> tuple[faiss.Index, list[dict]]:
    domain_dir = VECTORSTORE_DIR / domain
    index = faiss.read_index(str(domain_dir / "faiss.index"))
    with (domain_dir / "metadata.jsonl").open(encoding="utf-8") as fh:
        metadata = [json.loads(line) for line in fh]
    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"[{domain}] index ({index.ntotal}) and metadata ({len(metadata)}) disagree - rebuild."
        )
    return index, metadata


def search(query: str, domain: str = DEFAULT_DOMAIN, top_k: int = TOP_K) -> list[dict]:
    """Return top-k chunks of a domain: {rank, chunk_id, file, page, score, text}."""
    index, metadata = _ready(domain)
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
