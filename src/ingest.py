"""Chunk -> embed -> FAISS. Run once to build vectorstore/ (fast, a few seconds).

Design notes kept deliberately simple:
- Chunks are cut per page, so page metadata stays honest.
- TextSplitter(RecursiveCharacterTextSplitter) with overlap keeps sentences
  that span chunk boundaries readable in both chunks.
- Embeddings are L2-normalised and searched with IndexFlatIP => cosine similarity.
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


def build_index() -> dict:
    t0 = time.perf_counter()
    pages, loaded, skipped = load_directory(RAW_DIR)
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError("Ingestion produced 0 chunks - check data/raw contents.")

    print(f"[ingest] files:      {loaded}")
    if skipped:
        print(f"[ingest] SKIPPED:    {skipped}")
    print(f"[ingest] pages:      {len(pages)}")
    print(f"[ingest] chunks:     {len(chunks)}")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )

    import faiss
    import numpy as np

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype="float32"))

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTORSTORE_DIR / "faiss.index"))

    with (VECTORSTORE_DIR / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for chunk_id, chunk in enumerate(chunks):
            record = {"chunk_id": chunk_id, **chunk}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
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
    (VECTORSTORE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ingest] vectorstore ready in {manifest['embed_seconds']}s -> {VECTORSTORE_DIR}")
    return manifest


if __name__ == "__main__":
    build_index()
