"""Retrieval hit-rate evaluation on a hand-written demo Q/A set.

Metric (retrieval, model-free and reproducible):
- file hit:    expected source file appears among the top-k retrieved chunks
- page hit:    the exact expected page appears among the top-k retrieved chunks
- top1 score:  similarity of the best chunk (sanity check on match quality)

Label: this is a *rebuild demo set* we wrote for this repo on public course
notes. It is not a benchmark of the Clinical RAG project on the CV.

Usage:  python -m eval.run_eval [--k 4] [--no-llm]
        --no-llm measures retrieval only (fast, no Ollama needed)
Metrics are written to eval/eval_results.md.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from src.config import TOP_K, VECTORSTORE_DIR
from src.retrieve import search

QA_PATH = Path(__file__).parent / "qa_pairs.csv"
RESULTS_MD = Path(__file__).parent / "eval_results.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--no-llm", action="store_true",
                        help="retrieval metrics only, skip answer generation")
    args = parser.parse_args()

    with QA_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not (VECTORSTORE_DIR / "faiss.index").exists():
        raise SystemExit("No index - run `python -m src.ingest` first.")

    # expected_page is optional; without it only the file-level rate is reported
    has_pages = "expected_page" in rows[0] and rows[0].get("expected_page")

    file_hits = page_hits = 0
    n_page_scored = 0
    results = []
    for row in rows:
        hits = search(row["question"], top_k=args.k)
        file_hit = any(h["file"] == row["expected_source"] for h in hits)
        file_hits += file_hit
        page_hit = None
        expected_pair = None
        if has_pages:
            expected_page = int(row["expected_page"])
            expected_pair = next(
                (h for h in hits
                 if h["file"] == row["expected_source"] and h["page"] == expected_page),
                None,
            )
            if expected_page > 0:  # files without real pages (page 0) are not scored
                n_page_scored += 1
                page_hit = expected_pair is not None
                page_hits += bool(page_hit)
        results.append({
            "id": row["id"],
            "question": row["question"],
            "expected_source": row["expected_source"],
            "expected_page": row.get("expected_page"),
            "retrieved_files": [h["file"] for h in hits],
            "retrieved_pages": [(h["file"], h["page"]) for h in hits[:2]],
            "file_hit": file_hit,
            "page_hit": page_hit,
            "top1_score": round(hits[0]["score"], 3) if hits else 0.0,
        })

    n = len(results)
    file_rate = 100.0 * file_hits / n
    page_line = f"page hit rate : {page_hits}/{n_page_scored} = {100.0 * page_hits / n_page_scored:.0f}%" \
        if n_page_scored else "page hit rate : not scored (missing expected_page)"

    print(f"\n=== Retrieval eval on {n} demo questions (k={args.k}) ===")
    for r in results:
        flag = "OK " if r["file_hit"] else "MISS"
        print(f"[{flag}] {r['id']}: {r['question'][:60]:<60} top1={r['top1_score']}")
    print(f"file hit rate : {file_hits}/{n} = {file_rate:.0f}%")
    print(page_line)

    if not args.no_llm:
        from src.chat import ask_question
        print("\n=== Spot-check answers (first 5 questions, with LLM) ===")
        for r in results[:5]:
            try:
                outcome = ask_question(r["question"])
                print(f"\n{r['id']}: {outcome['answer'][:200]}")
                print(f"   cited: {[s['file'] for s in outcome['sources']]}")
            except Exception as exc:
                print(f"\n{r['id']}: LLM spot-check failed: {exc}")

    lines = [
        "# Retrieval eval — rebuild demo set",
        "",
        f"- Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- QA set: {n} hand-written questions on public course notes (this repo only)",
        f"- top_k: {args.k} · embeddings: see vectorstore/manifest.json",
        "",
        "| metric | hits | rate |",
        "|---|---|---|",
        f"| file hit rate | {file_hits}/{n} | {file_rate:.0f}% |",
    ]
    if n_page_scored:
        lines.append(f"| page hit rate | {page_hits}/{n_page_scored} | {100.0 * page_hits / n_page_scored:.0f}% |")
    lines += [
        "",
        "## Page-level misses",
        "",
        *(f"- {r['id']}: {r['question']}"
          for r in results if r.get("page_hit") is False),
        "",
        "Page-level metric: expected_page must be among the top-k retrieved chunks. "
        "Questions whose answer text spans more than one adjacent page often stick "
        "to the neighbouring page; the file-level metric above is the primary signal.",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nresults written to {RESULTS_MD}")


if __name__ == "__main__":
    main()
