"""Domain RAG chatbot - single-page Streamlit UI.

Left: chat with citations. Right (expander): the sources behind the answer.
A radio in the sidebar switches domain; each domain has its own FAISS index.
Run: streamlit run app.py  (build the indexes first: python -m src.ingest)
"""
from __future__ import annotations

import os

# The embedder is already in the local HF cache after `python -m src.ingest`.
# Running offline removes hub network checks from the serve path: fully
# offline-friendly AND immune to slow/blocked venue WiFi stalls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
VSTORE = PROJECT_ROOT / "vectorstore"

DOMAIN_LABELS = {
    "course": "🎓 Course notes",
    "medical": "🩺 Medical literature",
}
DOMAIN_PROMPTS = {
    "course": "Ask about the course notes…",
    "medical": "Ask about the medical abstracts and patient pages…",
}

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="🎓", layout="centered")


def available_domains() -> list[str]:
    if not VSTORE.is_dir():
        return []
    return sorted(
        d.name for d in VSTORE.iterdir()
        if d.is_dir() and (d / "faiss.index").exists()
    )


DOMAINS = available_domains()
if not DOMAINS:
    st.error("No indexes found in `vectorstore/`. Build them first:\n\n`python -m src.ingest`")
    st.stop()

if "domain" not in st.session_state or st.session_state.domain not in DOMAINS:
    st.session_state.domain = DOMAINS[0]

manifest = json.loads(
    (VSTORE / st.session_state.domain / "manifest.json").read_text(encoding="utf-8")
)

# --- sidebar: domain switch, corpus, status ---
with st.sidebar:
    st.title("📚 Domain RAG")
    st.caption("Retrieval-augmented answers with citations — one index per domain.")

    st.subheader("Domain")
    chosen = st.radio(
        "Answering from:",
        DOMAINS,
        format_func=lambda d: DOMAIN_LABELS.get(d, d),
        label_visibility="collapsed",
    )
    if chosen != st.session_state.domain:
        st.session_state.domain = chosen
        st.session_state.messages = []  # each domain keeps its own conversation

    st.subheader("Indexed corpus")
    for name, n in manifest["per_file"].items():
        st.markdown(f"- `{name}` — {n} chunks")
    st.caption(
        f"{manifest['n_files']} files · {manifest['n_chunks']} chunks "
        f"(size {manifest['chunk_size']}, overlap {manifest['chunk_overlap']})\n\n"
        f"Embeddings: `{manifest['embed_model']}`"
    )

    from src.chat import ollama_reachable
    from src.config import OLLAMA_MODEL, TOP_K

    if ollama_reachable():
        st.success(f"LLM online · `{OLLAMA_MODEL}` · top_k={TOP_K}")
    else:
        st.error(
            "Ollama unreachable. Start it with `ollama serve` "
            "(or the desktop app) and reload."
        )

    st.divider()
    st.caption(
        "Rebuild demo on public documents (course notes, PubMed abstracts, "
        "MedlinePlus pages) — not the CV Clinical-RAG project. Answers come only "
        "from the selected domain's index, else the bot says so."
    )

# --- chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    """Render the sources that fed the last answer."""
    rows = []
    for s in sources:
        page = "whole file" if s["page"] == 0 else f"p.{s['page']}"
        rows.append(f"| [{s['cite']}] | `{s['file']}` | {page} | {s['score']} |")
    st.expander("🔎 Sources used").markdown(
        "| cite | file | page | similarity |\n|---|---|---|---|\n" + "\n".join(rows)
    )


for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if user_question := st.chat_input(DOMAIN_PROMPTS.get(st.session_state.domain, "Ask…")):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages…"):
            try:
                from src.chat import ask_question

                result = ask_question(user_question, domain=st.session_state.domain)
            except Exception as exc:
                result = None
                st.error(f"Something failed: {exc}")

        if result:
            answer = result["answer"]
            st.markdown(answer)
            if result["sources"]:
                render_sources(result["sources"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": result["sources"]}
                )
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Pipeline: PDF/DOCX/TXT → chunks (900/150) · MiniLM-L6-v2 · FAISS per domain · "
    "Ollama llama3.2:3b, temperature 0, citation-only prompting."
)
