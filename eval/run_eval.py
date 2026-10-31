"""Retrieval hit-rate evaluation on a hand-written demo Q/A set.

Metric (retrieval, model-free and reproducible):
- file hit:    expected source file appears among the top-k retrieved chunks
- page hit:    the exact expected page appears among the top-k retrieved chunks
- top1 score:  similarity of the best chunk (sanity check on match quality)

Label: this is a *rebuild demo set* we wrote for this repo on public course
notes. It is not a benchmark of the Clinical RAG project on the CV.

Usage:  python -m eval.run_eval [--k 6] [--no-llm]
        --no-llm measures retrieval only (fast, no Ollama needed)
Metrics are written to eval/eval_results.md (per-domain breakdown included).
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
    from src.retrieve import available_domains

    if not available_domains():
        raise SystemExit("No indexes - run `python -m src.ingest` first.")

    # expected_page is optional per row; rows with 0 (whole-file docs) are not page-scored
    domains = sorted({r.get("domain", "course") for r in rows})
    counts = {d: {"files": 0, "pages": 0, "n": 0, "n_page": 0} for d in domains}

    results = []
    for row in rows:
        dom = row.get("domain", "course")
        hits = search(row["question"], domain=dom, top_k=args.k)
        file_hit = any(h["file"] == row["expected_source"] for h in hits)
        page_hit = None
        if row.get("expected_page") and int(row["expected_page"]) > 0:
            expected_page = int(row["expected_page"])
            page_hit = any(
                h["file"] == row["expected_source"] and h["page"] == expected_page
                for h in hits
            )
            counts[dom]["n_page"] += 1
            counts[dom]["pages"] += bool(page_hit)
        counts[dom]["files"] += file_hit
        counts[dom]["n"] += 1
        results.append({
            "id": row["id"],
            "domain": dom,
            "question": row["question"],
            "expected_source": row["expected_source"],
            "expected_page": row.get("expected_page"),
            "retrieved_files": [h["file"] for h in hits[:2]],
            "file_hit": file_hit,
            "page_hit": page_hit,
            "top1_score": round(hits[0]["score"], 3) if hits else 0.0,
        })

    n = len(results)
    file_hits = sum(r["file_hit"] for r in results)
    n_page_scored = sum(c["n_page"] for c in counts.values())
    page_hits = sum(c["pages"] for c in counts.values())
    file_rate = 100.0 * file_hits / n

    print(f"\n=== Retrieval eval on {n} demo questions (k={args.k}) ===")
    for r in results:
        flag = "OK " if r["file_hit"] else "MISS"
        print(f"[{flag}] {r['id']} ({r['domain'][:7]}): {r['question'][:48]:<48} top1={r['top1_score']}")
    print(f"file hit rate overall : {file_hits}/{n} = {file_rate:.0f}%")
    if n_page_scored:
        print(f"page hit rate         : {page_hits}/{n_page_scored} = {100.0 * page_hits / n_page_scored:.0f}%")
    for d, c in counts.items():
        f_rate = 100.0 * c["files"] / c["n"] if c["n"] else 0
        page_str = (
            f", page {c['pages']}/{c['n_page']} ({100.0 * c['pages'] / c['n_page']:.0f}%)"
            if c["n_page"] else ""
        )
        print(f"  domain '{d}': file {c['files']}/{c['n']} ({f_rate:.0f}%){page_str}")

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
        f"- QA set: {n} hand-written questions on public documents, this repo only",
        f"  (course notes + PubMed abstracts / MedlinePlus pages — NOT CV project data)",
        f"- top_k: {args.k} · embeddings: see vectorstore/*/manifest.json",
        "",
        "| metric | hits | rate |",
        "|---|---|---|",
        f"| file hit rate (overall) | {file_hits}/{n} | {file_rate:.0f}% |",
    ]
    if n_page_scored:
        lines.append(f"| page hit rate (pages exist) | {page_hits}/{n_page_scored} | {100.0 * page_hits / n_page_scored:.0f}% |")
    lines += [
        "",
        "## Per domain",
        "",
        "| domain | file hits | rate |",
        "|---|---|---|",
        *(f"| {d} | {c['files']}/{c['n']} | {100.0 * c['files'] / c['n']:.0f}% |" for d, c in counts.items()),
        "",
        "## Page-level misses (course domain)",
        "",
        *(f"- {r['id']}: {r['question']}" for r in results if r.get("page_hit") is False),
        "",
        "Page-level metric: expected_page must be among the top-k retrieved chunks "
        "(only scored where real pages exist, i.e. PDFs). Questions whose answer "
        "text spans more than one adjacent page often stick to the neighbouring "
        "page; the file-level metric above is the primary signal.",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nresults written to {RESULTS_MD}")


if __name__ == "__main__":
    main()
