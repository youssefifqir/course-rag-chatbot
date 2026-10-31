# Course & Medical RAG Chatbot 🎓🩺

A **retrieval-augmented generation (RAG)** chatbot that answers questions
**strictly from the documents it was indexed on**, with page-level citations,
and states clearly when an answer is not in the material.

One engine, two domain modules — each domain gets its own FAISS index, selected
from the sidebar:

- **🎓 Course notes** — ML, Python and NLP/RAG course texts
- **🩺 Medical literature** — PubMed abstracts + NIH MedlinePlus patient pages

Everything runs locally: public documents downloaded on first run, Ollama for
the LLM. Once the assets are cached, the entire stack works **fully offline**.

> 📝 **Scope:** this is a self-contained implementation built from scratch on
> public data. It is independent of any other system described elsewhere, and
> every metric reported here was measured on this repository's own evaluation
> suite.

![demo](docs/screenshot_demo.png)

## How it works

| Stage | Where | Detail |
|---|---|---|
| Load | `src/loaders.py` | PDF / DOCX / TXT → page-level text (page numbers preserved) |
| Chunk | `src/ingest.py` | `RecursiveCharacterTextSplitter`, 900 chars, 150 overlap, per-page |
| Embed | `sentence-transformers` | `all-MiniLM-L6-v2`, L2-normalised |
| Index | FAISS | `IndexFlatIP` (exact search = cosine similarity), one index per domain |
| Retrieve | `src/retrieve.py` | top-k=6, similarity metadata kept |
| Condense | `src/answer.py` | later turns are rewritten into a standalone query for retrieval (few-shot, temp 0) — the equivalent of `create_history_aware_retriever` |
| Gate | `src/chat.py` | similarity < 0.30 → "not found" without an answer call |
| Answer | `src/answer.py` | Ollama `llama3.2:3b`, temperature 0, context-only prompt with `[n]` citations |
| UI | `app.py` | Streamlit: domain switch, chat, sources table (file, page, similarity) |

## Quickstart

```bash
# 1) environment
python -m venv .venv
.venv\Scripts\activate            # Windows; "source .venv/bin/activate" on Linux/macOS
pip install -r requirements.txt

# 2) install Ollama (https://ollama.com) and pull a small model
ollama pull llama3.2:3b

# 3) download the public course corpus; optionally the medical corpus
bash scripts/download_corpus.sh
python scripts/download_medical_corpus.py

# 4) build one FAISS index per domain (~2-3 min on CPU)
python -m src.ingest

# 5) preload the model once so the first answer is fast (~10 s)
ollama run llama3.2:3b "Say OK"   # the app then keeps it resident for 30 min

# 6) run the app
streamlit run app.py              # opens http://localhost:8501
```

## The corpora (all public, free to use for study)

| File | Source | License / attribution |
|---|---|---|
| `ciml_machine_learning.pdf` | *A Course in Machine Learning* v0.99, Hal Daumé III ([ciml.info](https://ciml.info)) | free online edition, © author |
| `think_python.pdf` | *Think Python* 2e, Allen B. Downey ([greenteapress.com](https://greenteapress.com/wp/think-python-2e/)) | CC BY-NC |
| `slp3_ch7_transformers.pdf`, `slp3_ch9_masked_lm.pdf`, `slp3_ch11_ir_rag.pdf` | *Speech and Language Processing* 3rd ed. draft chapters 7/9/11, Jurafsky & Martin ([web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)) | free draft for coursework |

### Medical corpus (all public / public-domain)

| Type | What | Source |
|---|---|---|
| ~20 abstracts | semaglutide (STEP trials), SGLT2 inhibitors in HFpEF, metformin, inhaler technique | [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25499/) (title/authors/abstract only, PMID kept) |
| 4 patient pages | diabetes, hypertension, asthma, obesity | [NIH MedlinePlus](https://medlineplus.gov/) (US Gov public domain) |

Fetched by `scripts/download_medical_corpus.py`. No credentials or restricted
datasets involved.

## Evaluation — 30-question curated retrieval set

`eval/qa_pairs.csv` contains 30 questions authored for this repository — 20 on
the course corpus (file and page verified manually) and 10 on the medical
corpus — each with its expected source. `python -m eval.run_eval --no-llm`
measures retrieval quality deterministically (no LLM involved):

| scope | file hit rate | page hit rate |
|---|---|---|
| **overall (30 Q)** | **30/30 (100%)** | — |
| 🎓 course (20 Q) | 20/20 (100%) | 15/20 (75%) |
| 🩺 medical (10 Q) | 10/10 (100%) | not scored (whole-file docs) |

- **File-level** is the primary metric; **page-level** is the strict course-only
  variant — the exact page must appear among the top-k retrieved chunks. The 5
  page-misses are questions whose answer text spans adjacent pages and
  retrieval keeps the neighbouring page.
- `k=6` vs `k=4`: page-level recall rose from 65% to 75%; the cost is a longer
  prompt. `k=6` was selected after measuring both.
- The similarity gate (0.30) follows from a measured margin: in-corpus
  questions score ≥ ~0.44, off-corpus probes ~0.19–0.21.
- `python -m eval.run_eval` (with LLM) additionally spot-checks five answers
  end-to-end.

Results are written to `eval/eval_results.md` and committed.

## Offline operation

Once `data/raw/`, `vectorstore/`, the embedder (HF cache) and the Ollama model
are on disk, the stack runs with no network access:

```bash
HF_HUB_OFFLINE=1 streamlit run app.py
```

The app enables offline mode by default (it fails to cached assets instead of
touching the Hub). `eval/eval_results.md`, the screenshots in `docs/` and
`docs/demo_recording.webm` (a recorded live session) are committed as
reproducible artifacts.

## Design notes

- **RAG over fine-tuning**: the corpus is per-deployment and changes; a
  fine-tune would bake a fixed dataset into the weights and still hallucinate.
  RAG keeps knowledge inspectable (you see exactly which chunk an answer came
  from), updatable by re-running `python -m src.ingest`, and auditable
  (page citations).
- **One engine, per-domain indexes**: each `data/raw/<domain>/` folder builds
  its own FAISS index, so course questions never cross-search PubMed abstracts
  and vice versa. Adding a domain is a folder plus an ingest run — no code
  changes.
- **History-aware retrieval**: vague follow-ups such as *"could you detail it
  more"* carry no topical keywords; embedded raw, they retrieve unrelated
  pages. A condense step rewrites the last exchanges plus the new question
  (LLM, temperature 0, few-shot prompt) into one standalone query *before*
  retrieval, and the rewritten query is displayed in the sources panel. This is
  the equivalent of LangChain's `create_history_aware_retriever`, implemented
  directly over the FAISS search to keep page metadata under our control.
- **Hallucination control**: the prompt answers *only* from the numbered
  context passages and refuses otherwise; temperature 0; a measured similarity
  gate rejects off-corpus questions before the LLM is ever called. Small local
  models occasionally refuse even answerable questions — refusing over guessing
  is treated as the safer failure mode.
- **Chunk size 900 / overlap 150**: ~1-2 paragraphs per chunk — large enough to
  keep a definition (usually 3-5 sentences) in one piece, small enough that 6
  chunks fit well under the 3B model's context window. Overlap keeps sentences
  that straddle boundaries readable in both chunks.
- **LangChain + direct FAISS**: LangChain for splitting and chat orchestration;
  embeddings + FAISS is an explicit implementation so page-level metadata is
  preserved exactly — generic wrappers make that harder for this use case.
- **Local vs hosted LLM**: a locally served model keeps data on the machine,
  costs nothing per token and works offline. The code reads `OLLAMA_BASE_URL`,
  so pointing it at a hosted endpoint is a configuration change, not a code
  change.
- **Roadmap**: per-file retrieval filters within a domain, embedding
  fine-tuning, cross-encoder reranking before the LLM, token streaming, and a
  larger hosted model for a local-vs-hosted comparison.

## Project layout

```
app.py                  Streamlit UI (domain switch, chat, sources)
src/
  config.py             all knobs (env-overridable)
  loaders.py            PDF/DOCX/TXT -> pages
  ingest.py             chunk -> embed -> FAISS, one index per domain
  retrieve.py           search + metadata (domain-aware)
  answer.py             prompt + Ollama call
  chat.py               question -> answer pipeline (+ similarity gate)
eval/
  qa_pairs.csv          30 curated Q/A across both domains (with expected sources)
  run_eval.py           retrieval hit-rate report -> eval_results.md
  eval_results.md       latest measured results (committed)
scripts/
  download_corpus.sh    fetch public course PDFs + verify
  download_medical_corpus.py  fetch PubMed abstracts + MedlinePlus pages
  probe_retrieval.py    utility to inspect retrieval placement
  take_screenshots.py   utility to regenerate screenshots + demo recording
docs/                   screenshots + recorded demo
Dockerfile, docker-compose.yml
```

## FAQ

- *"Why does the first answer take a while?"* — Ollama loads the model on the
  first request. Preload it once (`ollama run llama3.2:3b "Say OK"`); the app
  then keeps it resident for 30 minutes between questions (`keep_alive`).
- *"Wrong page in a citation?"* — retrieval sometimes keeps the chunk on the
  neighbouring page; the sources panel exists so the answer's evidence is
  always visible.
- *"What if I ask something not in the notes?"* — it answers that it could not
  find the information and shows the (low) similarity scores for transparency.
- *"What happens with vague follow-ups like 'detail it more'?"* — the last
  turns plus the new question are rewritten into a standalone search query
  before retrieval (shown live in the sources panel: "searched as: …"), and the
  answer prompt also receives recent conversation.
- *"Does it give medical advice?"* — no. Answers are grounded in public
  abstracts and patient-information pages, cite their sources, and the gate
  refuses anything outside the indexed documents.
- *"Why do some screenshots show refusals?"* — to demonstrate the controlled
  failure modes: the off-corpus refusal and the occasionally over-conservative
  behaviour of a 3B model. Both are documented behavior rather than a hidden
  flaw.

## License

MIT — see [LICENSE](LICENSE).
