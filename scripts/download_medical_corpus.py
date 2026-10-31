"""Fetch the public medical corpus into data/raw/medical/.

Sources (all public domain / freely readable):
- PubMed abstracts via NCBI E-utilities (esearch + efetch, XML), a handful of
  abstracts per topic (~20 total). Only title/authors/abstract text are stored,
  with the PMID for attribution.
- NIH MedlinePlus condition overview pages (plain-language patient info).

Run:  python scripts/download_medical_corpus.py
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "data" / "raw" / "medical"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = "youssef-ifqir-demo@example.com"   # NCBI asks callers to identify themselves

TOPICS = [
    ("semaglutide obesity treatment", "semaglutide"),
    ("SGLT2 inhibitors heart failure", "sglt2_heart_failure"),
    ("metformin type 2 diabetes", "metformin"),
    ("asthma treatment inhalers", "asthma_therapy"),
]

MEDLINEPLUS = [
    ("https://medlineplus.gov/highbloodpressure.html", "medlineplus_hypertension.txt"),
    ("https://medlineplus.gov/diabetestype2.html", "medlineplus_diabetestype2.txt"),
    ("https://medlineplus.gov/asthma.html", "medlineplus_asthma.txt"),
    ("https://medlineplus.gov/obesity.html", "medlineplus_obesity.txt"),
]


def http_get(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"rag-demo/0.1 ({EMAIL})"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_pubmed(query: str, stem: str, n: int = 5) -> int:
    import json

    args = urllib.parse.urlencode({
        "db": "pubmed", "term": query, "retmax": str(n),
        "retmode": "json", "sort": "relevance", "email": EMAIL,
    })
    data = http_get(f"{EUTILS}/esearch.fcgi?{args}")
    found = json.loads(data).get("esearchresult", {}).get("idlist", [])
    if not found:
        print(f"[skip] no PMIDs for '{query}'")
        return 0
    time.sleep(0.5)
    fetch_args = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(found), "retmode": "xml", "email": EMAIL,
    })
    xml_data = http_get(f"{EUTILS}/efetch.fcgi?{fetch_args}")
    root = ET.fromstring(xml_data)
    saved = 0
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = (article.findtext(".//ArticleTitle") or "").strip()
        abstract = " ".join(
            (node.text or "").strip()
            for node in article.findall(".//Abstract/AbstractText")
        ).strip()
        journal = (article.findtext(".//Journal/Title") or "").strip()
        year = (article.findtext(".//JournalIssue/PubDate/Year")
                or article.findtext(".//Article/ArticleDate/Year") or "").strip()
        authors = ", ".join(
            (a.findtext("LastName") or "") + " " + (a.findtext("Initials") or "")
            for a in article.findall(".//Author")[:4]
        )
        if not pmid or not abstract:
            continue
        text = (
            f"{title}\n"
            f"Authors: {authors}\nJournal: {journal} ({year}) — PubMed PMID: {pmid}\n"
            f"\nAbstract: {abstract}\n"
        )
        out = OUT / f"pubmed_{stem}_{pmid}.txt"
        out.write_text(text, encoding="utf-8")
        saved += 1
    print(f"[ok] {stem}: saved {saved} abstracts for '{query}'")
    return saved


class _TextExtract(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "nav", "footer", "header"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "nav", "footer", "header"}:
            self.skip_depth = max(0, self.skip_depth - 1)
        if tag in {"p", "li", "h1", "h2", "h3", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip() + " ")


def fetch_medlineplus(url: str, name: str) -> bool:
    raw = http_get(url).decode("utf-8", errors="ignore")
    parser = _TextExtract()
    parser.feed(raw)
    text = re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts))
    if len(text) < 500:
        print(f"[skip] {name}: page too short after extraction ({len(text)} chars)")
        return False
    (OUT / name).write_text(
        f"MedlinePlus (NIH) — {url}\nPublic domain patient information page.\n\n{text}",
        encoding="utf-8",
    )
    print(f"[ok] {name}: {len(text)} chars")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for query, stem in TOPICS:
        total += fetch_pubmed(query, stem)
        time.sleep(0.5)
    for url, name in MEDLINEPLUS:
        total += 1 if fetch_medlineplus(url, name) else 0
        time.sleep(0.5)
    print(f"done: {total} medical documents in {OUT}")


if __name__ == "__main__":
    main()
