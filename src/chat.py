"""One entrypoint shared by the Streamlit UI and the eval script:
question -> retrieved sources -> LLM answer (+ honest 'not found' handling).
"""
from __future__ import annotations

from src.config import OLLAMA_BASE_URL


class NotIndexable(RuntimeError):
    pass

def ask_question(question: str) -> dict:
    """Returns {'answer': str, 'sources': [compressed hits], 'retrieved': hits}."""
    from src.answer import answer_question
    from src.config import SIM_GATE
    from src.retrieve import search

    if not question or not question.strip():
        raise ValueError("Empty question.")

    hits = search(question.strip())
    not_found = "I could not find this in the indexed course material."
    if not hits or hits[0]["score"] < SIM_GATE:
        # Off-corpus question: refuse before involving the LLM (hallucination control)
        return {
            "answer": not_found,
            "sources": [],
            "retrieved": hits,
        }

    compact = [
        {"cite": h["rank"], "file": h["file"], "page": h["page"], "score": round(h["score"], 3)}
        for h in hits
    ]
    try:
        answer = answer_question(question.strip(), hits)
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    return {"answer": answer, "sources": compact, "retrieved": hits}


def ollama_reachable() -> bool:
    """Cheap connectivity check used by the UI status pill."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False
