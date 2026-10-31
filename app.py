"""Course RAG chatbot - single-page Streamlit UI.

Left: chat with citations. Right (expander): the sources behind the answer.
Run: streamlit run app.py  (build the index first: python -m src.ingest)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
VSTORE = PROJECT_ROOT / "vectorstore"

st.set_page_config(page_title="Course RAG Chatbot", page_icon="🎓", layout="centered")

# --- index present? ---
if not (VSTORE / "faiss.index").exists():
    st.error(
        "No index found in `vectorstore/`. Build it first:\n\n"
        "`python -m src.ingest`"
    )
    st.stop()

manifest = json.loads((VSTORE / "manifest.json").read_text(encoding="utf-8"))

# --- sidebar: what's indexed / status ---
with st.sidebar:
    st.title("📚 Course RAG")
    st.caption("Retrieval-augmented answers from course notes, with citations.")

    st.subheader("Indexed corpus")
    for name, n in manifest["per_file"].items():
        st.markdown(f"- `{name}` — {n} chunks")
    st.caption(
        f"{manifest['n_pages']} pages → {manifest['n_chunks']} chunks "
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
        "Rebuild demo on public course notes — not the CV Clinical-RAG project. "
        "Answers come only from the indexed PDFs, else the bot says so."
    )

# --- chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []


def seed_from_url() -> None:
    """Support demo links: http://localhost:8501/?q=What+is+RAG
    asks the question once on load - nice for sharing a one-shot demo link."""
    params = st.query_params
    q = params.get("q")
    if q and not st.session_state.messages:
        try:
            from src.chat import ask_question

            result = ask_question(q)
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
        except Exception as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Seed question failed: {e}"}
            )
        params.clear()


seed_from_url()


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

if user_question := st.chat_input("Ask about the indexed course notes…"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages…"):
            try:
                from src.chat import ask_question

                result = ask_question(user_question)
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
    "Pipeline: PDF → chunks (900/150) · MiniLM-L6-v2 FAISS index · Ollama llama3.2:3b, "
    "temperature 0, citation-only prompting."
)
