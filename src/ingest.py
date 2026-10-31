"""Chunk -> embed -> FAISS, one index per domain. Run once to build vectorstore/.

Each subdirectory of data/raw is a domain (course, medical, ...). Domains get
SEPARATE FAISS indexes so retrieval never mixes course notes with, say, PubMed
abstracts — same engine, selective memory. Per-domain artifacts:

    vectorstore/<domain>/faiss.index
    vectorstore/<domain>/metadata.jsonl
    vectorstore/<domain>/manifest.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    RAW_DIR,
    VECTORSTORE_DIR,
)
from src.loaders import load_directory


def chunk_pages(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            piece = piece.strip()
            if len(piece) < 60:  # drop headings-only / caption-only fragments
                continue
            chunks.append({"text": piece, "metadata": dict(page["metadata"])})
    return chunks


def build_domain_index(domain: str, model: SentenceTransformer) -> dict:
    t0 = time.perf_counter()
    pages, loaded, skipped = load_directory(RAW_DIR / domain)
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError(f"Domain '{domain}': ingestion produced 0 chunks - check contents.")

    print(f"[ingest:{domain}] files:      {loaded}")
    if skipped:
        print(f"[ingest:{domain}] SKIPPED:    {skipped}")
    print(f"[ingest:{domain}] pages:      {len(pages)} · chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )

    import faiss
    import numpy as np

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype="float32"))

    domain_dir = VECTORSTORE_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(domain_dir / "faiss.index"))
    with (domain_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for chunk_id, chunk in enumerate(chunks):
            fh.write(json.dumps({"chunk_id": chunk_id, **chunk}, ensure_ascii=False) + "\n")

    manifest = {
        "domain": domain,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_files": len(loaded),
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "per_file": {
            name: sum(1 for c in chunks if c["metadata"]["source"] == name)
            for name in sorted({c["metadata"]["source"] for c in chunks})
        },
        "embed_seconds": round(time.perf_counter() - t0, 1),
    }
    (domain_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ingest:{domain}] ready in {manifest['embed_seconds']}s -> {domain_dir}")
    return manifest


def build_all(domains: list[str] | None = None) -> dict[str, dict]:
    if domains is None:
        if not RAW_DIR.is_dir():
            raise SystemExit(
                f"{RAW_DIR} does not exist. Run scripts/download_corpus.sh and "
                "scripts/download_medical_corpus.py first."
            )
        domains = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    if not domains:
        raise SystemExit(f"No domain folders under {RAW_DIR} - nothing to ingest.")

    print(f"[ingest] domains: {domains}")
    model = SentenceTransformer(EMBED_MODEL)  # loaded once, reused per domain
    manifests = {d: build_domain_index(d, model) for d in domains}
    (VECTORSTORE_DIR / "manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifests


if __name__ == "__main__":
    build_all()
