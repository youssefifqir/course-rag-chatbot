"""One entrypoint shared by the Streamlit UI and the eval script:
question -> retrieved sources -> LLM answer (+ honest 'not found' handling).
"""
from __future__ import annotations

from src.config import OLLAMA_BASE_URL


class NotIndexable(RuntimeError):
    pass

def ask_question(question: str, domain: str = "course", history: list[dict] | None = None) -> dict:
    """Returns {'answer', 'sources', 'retrieved', 'rewritten'}.

    history = previous turns (user/assistant dicts) of THIS conversation.
    Follow-up questions get rewritten into a standalone query before retrieval.
    """
    from src.answer import answer_question, condense_question
    from src.config import SIM_GATE
    from src.retrieve import search

    if not question or not question.strip():
        raise ValueError("Empty question.")

    # 1) history-aware rewrite ("could you detail it more" -> standalone query)
    rewritten = condense_question(question.strip(), history)

    # 2) retrieval on the (possibly rewritten) query
    hits = search(rewritten, domain=domain)

    # 3) similarity floor: off-corpus question -> refuse without the answer LLM.
    #    NB: the rewrite itself does use the LLM when history exists.
    not_found = "I could not find this in the indexed documents for this domain."
    if not hits or hits[0]["score"] < SIM_GATE:
        return {
            "answer": not_found,
            "sources": [],
            "retrieved": hits,
            "rewritten": rewritten,
        }

    # 4) grounded answer, history passed for continuity
    compact = [
        {"cite": h["rank"], "file": h["file"], "page": h["page"], "score": round(h["score"], 3)}
        for h in hits
    ]
    try:
        answer = answer_question(question.strip(), hits, history=history)
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    return {"answer": answer, "sources": compact, "retrieved": hits, "rewritten": rewritten}


def ollama_reachable() -> bool:
    """Cheap connectivity check used by the UI status pill."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False
