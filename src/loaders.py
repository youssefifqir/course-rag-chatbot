"""Load PDF / DOCX / TXT files and return page-level text units.

Keeping page boundaries intact here means every chunk below keeps an honest
page number, which is what we show in the sources panel ("file, page N").
"""
from __future__ import annotations

from pathlib import Path

from src.config import SUPPORTED_EXTENSIONS

PageDocument = dict  # {"text": str, "metadata": {"source": str, "page": int}}


def _load_pdf(path: Path) -> list[PageDocument]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PageDocument] = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        if len(text.strip()) < 30:  # covers, blank pages, scanned images
            continue
        pages.append({"text": text, "metadata": {"source": path.name, "page": i}})
    return pages


def _load_docx(path: Path) -> list[PageDocument]:
    import docx  # python-docx

    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        return []
    # DOCX has no pages: treat the whole file as one unit, page 0 = "whole file"
    return [{"text": text, "metadata": {"source": path.name, "page": 0}}]


def _load_txt(path: Path) -> list[PageDocument]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    # TXT also has no real pagination -> one unit, page 0
    return [{"text": text, "metadata": {"source": path.name, "page": 0}}]


LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}


def load_directory(raw_dir: Path) -> tuple[list[PageDocument], list[str], list[str]]:
    """Load every supported file in raw_dir (one level).

    Each subdirectory of data/raw is a "domain" (course, medical, ...),
    managed by the caller; this function only ingests one."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Run scripts/download_corpus.sh first "
            "or drop your own PDFs/DOCX/TXT there."
        )

    all_pages: list[PageDocument] = []
    loaded, skipped = [], []
    for path in sorted(raw_dir.iterdir()):
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        try:
            pages = LOADERS[ext](path)
        except Exception as exc:  # one broken file must not kill ingestion
            skipped.append(f"{path.name}: {exc}")
            continue
        all_pages.extend(pages)
        loaded.append(f"{path.name} -> {len(pages)} page(s)")
    if not all_pages:
        raise RuntimeError(f"No readable files in {raw_dir}. Loaded: {loaded}")
    return all_pages, loaded, skipped
