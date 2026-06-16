"""
Configurações da aplicação.

Lê variáveis de ambiente com valores padrão para desenvolvimento local.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "arxiv_papers")

# --- Google Gemini ---
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# --- Hugging Face ---
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# Cadeia de fallback de modelos (do mais desejado ao mais leve)
GEMINI_MODELS: list[str] = [
    "gemma-4-27b-it",           # Gemma 4 (principal)
    "gemini-3.1-flash-lite",    # Fallback 1
    "gemini-2.5-flash-lite",    # Fallback 2
    "gemini-2.5-flash",         # Fallback 3 (último recurso)
]

# --- Snapshot ---
SNAPSHOT_URL: str = os.getenv(
    "SNAPSHOT_URL",
    "https://snapshots.qdrant.io/arxiv_abstracts-2108082541245612-2026-06-04-09-56-06.snapshot",
)

# --- Embedding ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "hkunlp/instructor-xl")
EMBEDDING_DIM: int = 768
