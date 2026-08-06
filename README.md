# Offline Multimodal RAG System

A fully offline Retrieval-Augmented Generation (RAG) system that ingests PDFs, Word documents, images, and audio files, stores them in a local vector database, and answers natural-language questions using a locally running LLM — no external API calls at query time.

---

## Features

- **Multimodal ingestion** — PDF (with scanned-page OCR fallback), DOCX, PNG/JPG (OCR), WAV/MP3 (Whisper transcription)
- **Fully offline** — all inference runs locally after the one-time model downloads
- **Semantic search** — sentence-transformers embeddings (`all-MiniLM-L6-v2`) stored in ChromaDB
- **Metadata filtering** — search can be scoped to a specific `file_type` or `source` filename
- **Local LLM** — Ollama integration; switch models from the UI without restarting
- **Voice input** — record a question with the browser microphone; transcribed by Whisper and shown for confirmation before submission
- **Grounded answers** — the LLM is instructed to answer only from retrieved context and says so when it cannot
- **Edge-case hardening** — empty files, oversized files, corrupt/password-protected PDFs, silent audio, and duplicate uploads all produce clear, actionable UI messages instead of crashes

---

## Project Structure

```
.
├── app/
│   ├── main.py               # Streamlit web application (full UI)
│   ├── rag_pipeline.py       # build_prompt() + generate_answer()
│   └── test_ollama.py        # Standalone Ollama connectivity check
├── ingestion/
│   ├── ingest.py             # Unified ingest_file() + validate_file()
│   ├── chunker.py            # Word-based overlapping chunker
│   ├── pdf_extractor.py      # PyMuPDF text + OCR-flag extraction
│   ├── docx_extractor.py     # python-docx paragraph + table extraction
│   ├── image_extractor.py    # pytesseract OCR extraction
│   └── audio_extractor.py    # faster-whisper transcription
├── retrieval/
│   ├── vector_store.py       # VectorStore class (store, search, reset)
│   └── tune_experiment.py    # Chunk-size/overlap comparison experiment
├── tests/
│   ├── test_edge_cases.py    # pytest suite (14 tests, all synthetic fixtures)
│   ├── check_persistence.py  # ChromaDB persistence proof (4 phases)
│   └── benchmark_search.py   # search() latency benchmark (20 queries)
├── models/                   # Reserved for future model utilities
├── data/                     # ChromaDB persists here (auto-created)
├── config.py                 # All configuration constants
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Tesseract OCR | any | Required for image and scanned-PDF ingestion |
| Ollama | latest | Required for answer generation |
| RAM | 4 GB min | 8 GB+ recommended for Llama3.1 8B |

---

## Setup

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

**Windows**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run it (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to `.env`:
   ```
   PYTESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

**macOS**
```bash
brew install tesseract
```

**Linux (Ubuntu / Debian)**
```bash
sudo apt-get update && sudo apt-get install tesseract-ocr
```

### 4. Install and start Ollama

1. Download from https://ollama.ai and install
2. Start the server:
   ```bash
   ollama serve
   ```
3. Pull at least one model:
   ```bash
   ollama pull phi3          # 3.8B — fast, low RAM
   ollama pull llama3.1:8b   # 8B  — better quality, needs 8 GB+ RAM
   ```

### 5. Create a `.env` file (optional)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3
PYTESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

All values fall back to the defaults in `config.py` if the file is absent.

---

## Running the App

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`.

### Workflow

1. **Upload** — drag files onto the upload panel; a 4-stage progress indicator shows validation → extraction → chunking → embedding
2. **Ask** — type a question or click **Start recording** to ask by voice
3. **Review** — the answer appears in a chat bubble; expand **Sources** to see the exact snippet and page/timestamp each sentence came from
4. **Switch models** — use the sidebar dropdown to change the active Ollama model without restarting
5. **Reset** — the sidebar **Reset Knowledge Base** button clears all stored chunks after a confirmation step

---

## Configuration

All constants live in `config.py`. Override sensitive values via `.env`.

| Key | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `phi3` | Default model (overridden by UI dropdown) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `CHUNK_SIZE` | `400` | Max words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping words between consecutive chunks |
| `CHROMA_DB_PATH` | `./data/chroma_db` | Persistent vector DB location |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_FILE_SIZE_MB` | `50` | Files larger than this are rejected at upload |
| `MIN_TRANSCRIPT_WORDS` | `3` | Audio transcripts shorter than this are treated as silence |
| `PYTESSERACT_PATH` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Windows Tesseract binary path |

---

## Supported File Formats

| Format | Extension(s) | Extraction method |
|---|---|---|
| PDF | `.pdf` | PyMuPDF; scanned pages fall back to Tesseract OCR |
| Word | `.docx` | python-docx (paragraphs + tables) |
| Image | `.png` `.jpg` `.jpeg` | Tesseract OCR with contrast pre-processing |
| Audio | `.wav` `.mp3` | faster-whisper (Whisper `base` model, CPU) |

---

## Module API Reference

### `ingestion/ingest.py`

```python
validate_file(file_path: str) -> None
# Raises FileNotFoundError or ValueError for: missing, empty, or oversized files.

ingest_file(file_path: str) -> list[dict]
# Returns: [{"text": str, "metadata": {"source", "file_type", "page_or_timestamp"}}]
# Raises ValueError for: corrupt/password-protected PDFs, silent audio, unsupported extensions.
```

### `ingestion/chunker.py`

```python
chunk_text(text: str, metadata: dict) -> list[dict]
# Splits text into overlapping word-based chunks.
# Returns: [{"text": str, "metadata": {..., "chunk_index": int}}]
```

### `retrieval/vector_store.py`

```python
store = VectorStore()

store.store_chunks(chunks: list[dict]) -> int
# Embeds and stores new chunks; skips exact duplicates (sha256 hash).
# Returns number of chunks actually written.

store.search(query, top_k=None, file_type=None, source=None) -> list[dict]
# Returns: [{"text", "metadata", "score"}] sorted by descending relevance.
# file_type / source apply ChromaDB where-clause filtering (not post-filtering).

store.count() -> int
store.reset() -> None
```

### `app/rag_pipeline.py`

```python
build_prompt(query: str, retrieved_chunks: list[dict]) -> str
# Constructs a grounded prompt with numbered citation blocks.

generate_answer(query, retrieved_chunks, model=None) -> dict
# Returns: {"answer": str, "sources": [{"source", "page_or_timestamp", "snippet"}]}
# Skips LLM call if retrieved_chunks is empty or query is blank.
```

---

## Tests

### Pytest edge-case suite

```bash
pytest tests/test_edge_cases.py -v
```

Covers (14 tests, all synthetic — no real files needed):
- Empty files (PDF, DOCX, WAV)
- Oversized file rejection
- Corrupt and password-protected PDF handling
- Silent / near-silent audio rejection
- Unsupported file extension
- Duplicate chunk deduplication
- Blank query guard in `generate_answer`

### Persistence check

Proves data survives a full ChromaDB client close/reopen cycle. Uses a temp directory — never touches the production DB.

```bash
python tests/check_persistence.py
```

### Search latency benchmark

Times `search()` across 20 queries and reports average, median, p95, min, max with an embed-vs-ChromaDB latency split.

```bash
python tests/benchmark_search.py           # 1 run per query (quick)
python tests/benchmark_search.py --runs 5  # 5 runs per query (stable averages)
```

### Ollama connectivity check

```bash
python app/test_ollama.py
```

### Retrieval tuning experiment

Compares chunk size/overlap configurations side by side to help pick optimal defaults.

```bash
python retrieval/tune_experiment.py
```

---

## System Requirements by Model

| Model | Parameters | Min RAM | Notes |
|---|---|---|---|
| `phi3` | 3.8B | 4 GB | Recommended default — fast |
| `llama3.1:8b` | 8B | 8 GB | Better quality, slower |
| `llama3.1:70b` | 70B | 32 GB | High-end systems only |

---

## Troubleshooting

**Ollama not reachable**
- Run `ollama serve` in a terminal and keep it open
- Verify: `curl http://localhost:11434/api/tags`
- Check the sidebar — it shows a warning with the exact command if Ollama is unreachable

**Model not found**
- Run `ollama pull <model-name>` (e.g. `ollama pull phi3`)
- The model name in the sidebar dropdown must match an installed model exactly

**Tesseract not found (Windows)**
- Confirm the install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- If installed elsewhere, set `PYTESSERACT_PATH` in `.env`

**File rejected — too large**
- Default limit is 50 MB; raise `MAX_FILE_SIZE_MB` in `config.py` if needed

**File rejected — no text extracted**
- PDFs: may be image-only without embedded text; OCR fallback runs automatically but requires Tesseract
- Images: low contrast or very small text reduces OCR accuracy
- Audio: recording may be too short or too quiet; `MIN_TRANSCRIPT_WORDS` threshold is 3 words

**Slow inference**
- Switch to a smaller model (`phi3` instead of `llama3.1:8b`)
- Reduce `CHUNK_SIZE` to send less context per prompt
- Close other memory-heavy applications

**Out of memory**
- Use `phi3` (4 GB RAM) instead of larger models
- Reduce `CHUNK_SIZE` and `TOP_K` in `config.py`

**ChromaDB errors on startup**
- Delete `./data/chroma_db/` and restart — the DB will be recreated empty
- Re-ingest your documents after clearing

---

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
