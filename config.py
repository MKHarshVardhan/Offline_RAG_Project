"""
Central configuration for the offline multimodal RAG system.
Override sensitive values via a .env file in the project root.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Ollama LLM ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3")

# --- Embeddings ---
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# --- Text chunking ---
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 50

# --- Vector store ---
CHROMA_DB_PATH: str = "./data/chroma_db"

# --- Retrieval ---
TOP_K: int = 5

# --- Ingestion limits ---
MAX_FILE_SIZE_MB: int = 50          # files larger than this are rejected
MIN_TRANSCRIPT_WORDS: int = 3       # audio transcripts shorter than this are treated as silence

# --- OCR ---
PYTESSERACT_PATH: str = os.getenv(
    "PYTESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
