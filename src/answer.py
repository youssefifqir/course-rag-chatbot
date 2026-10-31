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
    "1. Answer ONLY from the numbered context passages supplied with the question.\n"
    "2. Answer in 2-4 sentences that actually explain the idea in simple words;\n"
    "   never answer with a single quoted phrase on its own.\n"
    "3. Cite the passage numbers you used, like [2] or [1][3].\n"
    "4. If the passages do not contain the answer, say exactly: "
    '"I could not find this in the indexed documents for this domain." Do not guess.\n'
    "5. Use simple language, no invented definitions."
)

LLM = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
    num_predict=400,
    keep_alive="30m",  # don't unload the model between demo questions (default 5m)
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


def condense_question(question: str, history: list[dict] | None) -> str:
    """History-aware retrieval query rewrite (what LangChain's
    create_history_aware_retriever does against a BaseRetriever, implemented
    directly against our FAISS search).

    Follow-ups like "could you detail it more" carry no topic words, so the
    raw sentence embeds to noise. We take the last turns + the new question
    and let the LLM (temp 0) rewrite them into one standalone question; the
    rewritten query is what gets embedded and searched. Few-shot example kept
    in the prompt because llama3.2:3b is mediocre at zero-shot rewriting.
    """
    if not history:
        return question

    turns = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
        for m in list(history)[-4:]  # last 2-3 turns
    )
    message = (
        "Rewrite the follow-up question so it can be searched on its own.\n"
        "Rules:\n"
        '- Use the chat history to resolve pronouns and vague words ("it", "that", "more").\n'
        "- Keep the important keywords of the topic discussed.\n"
        "- Output ONLY the rewritten question, one line, no quotes.\n"
        "- If the question is already standalone, output it unchanged.\n\n"
        "Example:\n"
        "History:\n"
        "User: What is machine learning?\n"
        "Assistant: Machine learning is a field where algorithms learn patterns from "
        "data instead of being explicitly programmed [1].\n"
        "Follow-up: could you detail it more\n"
        "Rewritten: Explain in more detail what machine learning is and how it works\n\n"
        "Now rewrite:\n"
        f"History:\n{turns}\n"
        f"Follow-up: {question}\n"
        "Rewritten:"
    )
    try:
        out = LLM.invoke([("human", message)]).content.strip().strip('"').strip()
    except Exception:
        return question  # rewriting must never break answering
    if not out or "\n" in out or len(out) > 300:
        return question
    return out


def answer_question(question: str, hits: list[dict], history: list[dict] | None = None) -> str:
    context = build_context(hits)
    history_block = ""
    if history:
        turns = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in list(history)[-2:]
        )
        history_block = f"Recent conversation (for context):\n{turns}\n\n"
    message = (
        f"{history_block}"
        f"Question: {question}\n\n"
        f"Context passages from the documents:\n\n{context}\n\n"
        "Answer with citations, or say you could not find it in the material."
    )
    response = LLM.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", message),
        ]
    )
    return response.content.strip()
