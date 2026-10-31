"""LLM answer generation on top of retrieved chunks (Ollama local by default).

Hallucination control: the prompt only ever sees the retrieved chunks, must
answer exclusively from them with bracket citations, and must say "not found"
otherwise. Temperature 0.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama

from src.config import LLM_TEMPERATURE, MAX_CONTEXT_CHARS, OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = (
    "You are a patient teaching assistant for university students.\n"
    "Rules:\n"
    "1. Answer ONLY from the numbered context passages.\n"
    "2. Cite the passage numbers you used, like [2] or [1][3].\n"
    "3. If the passages do not contain the answer, say exactly: "
    '"I could not find this in the indexed course material." Do not guess.\n'
    "4. Be concise: 2-6 sentences, simple language, no invented definitions."
)

LLM = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
    num_predict=400,
)


def build_context(hits: list[dict]) -> str:
    """Numbered passages, truncated so the prompt stays well under the
    llama3.2 3B context window even at top_k=6."""
    blocks = []
    for hit in hits:
        text = hit["text"]
        if len(text) > MAX_CONTEXT_CHARS // len(hits):
            text = text[: MAX_CONTEXT_CHARS // len(hits)] + " [...]"
        page = hit["page"] if hit["page"] > 0 else "whole file"
        blocks.append(f'[{hit["rank"]}] ({hit["file"]}, p.{page})\n{text}')
    return "\n\n".join(blocks)


def answer_question(question: str, hits: list[dict]) -> str:
    context = build_context(hits)
    message = (
        f"Question: {question}\n\n"
        f"Context passages from the course material:\n\n{context}\n\n"
        "Answer with citations, or say you could not find it in the material."
    )
    response = LLM.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", message),
        ]
    )
    return response.content.strip()
