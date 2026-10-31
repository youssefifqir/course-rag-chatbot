"""Central configuration. Every knob can be overridden by an environment variable,
so the same code runs locally (Ollama) or with a hosted LLM later."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Paths ---
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = Path(os.environ.get("RAW_DIR", DATA_DIR / "raw"))
VECTORSTORE_DIR = Path(os.environ.get("VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore"))

# --- Retrieval ---
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))     # chars; ~1-2 short paragraphs
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))  # ~17% overlap so sentences at
                                                             # boundaries stay readable in both chunks
TOP_K = int(os.environ.get("TOP_K", "6"))

# Refusal gate: below this cosine similarity, the question is considered
# off-corpus and we answer "not found" without calling the LLM.
# 0.30 chosen from a measured gap: in-corpus questions score >= ~0.44,
# out-of-corpus probes scored ~0.19-0.21 in tests.
SIM_GATE = float(os.environ.get("SIM_GATE", "0.30"))

# --- LLM ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "6000"))

# Files of these types are ingested from data/raw
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
