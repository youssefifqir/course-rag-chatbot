"""Dev helper: for each eval question, print where retrieval lands.

Used once to fill expected_page in eval/qa_pairs.csv.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))  # run from anywhere

from src.config import TOP_K
from src.retrieve import search

QA = Path(__file__).parents[1] / "eval" / "qa_pairs.csv"

with QA.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        hits = search(row["question"], top_k=2)
        if not hits:
            print(f"{row['id']}: NO HITS")
            continue
        best = hits[0]
        pages = [f"{h['file']} p.{h['page']}" for h in hits]
        print(f"{row['id']} [{best['score']:.3f}] {row['question'][:48]:<48} -> {pages}")
