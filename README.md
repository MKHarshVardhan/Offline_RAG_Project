# Offline Multimodal RAG System

A fully offline Retrieval-Augmented Generation (RAG) system that processes multiple data formats (PDFs, Word documents, images, audio) and uses local LLMs for generation without any external API calls.

## Features

- **Multimodal Input**: Process PDFs, Word documents, images (OCR), and audio files
- **Fully Offline**: All processing runs locally after initial model downloads
- **Semantic Search**: Uses sentence transformers for embeddings
- **Vector Database**: ChromaDB for efficient retrieval
- **Local LLM**: Ollama integration with support for various models (Phi3, Llama3.1, etc.)
- **Web Interface**: Streamlit UI with voice input support

## Project Structure

```
.
├── ingestion/          # Document/data ingestion modules
├── retrieval/          # Retrieval and search modules
├── app/                # Streamlit web application
├── data/               # Data storage (documents, embeddings)
├── models/             # Model configuration and utilities
├── tests/              # Unit and integration tests
├── config.py           # Configuration constants
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Prerequisites

- Python 3.10 or higher
- Windows, macOS, or Linux
- ~8GB RAM minimum (more recommended for larger models)
- Tesseract OCR (for image text extraction)
- Ollama (for local LLM inference)

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (System Dependency)

**Windows:**
- Download the Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
- Add to your `.env` file:
  ```
  PYTESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
  ```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### 4. Install and Configure Ollama

1. Download Ollama from: https://ollama.ai
2. Install and run Ollama following the official instructions
3. The Ollama service will run in the background on `http://localhost:11434` by default

### 5. Download a Local Language Model

```bash
# Pull Phi3 (smaller, faster - recommended for most users)
ollama pull phi3

# OR pull Llama3.1 8B (larger, better quality - requires more resources)
ollama pull llama3.1:8b
```

You can switch between models by changing the `OLLAMA_MODEL` value in `config.py`.

### 6. Create `.env` File

Create a `.env` file in the project root (optional, for sensitive configs):

```env
OLLAMA_BASE_URL=http://localhost:11434
DEBUG=false
PYTESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Running the Application

### Start the Streamlit Web Interface

```bash
streamlit run app/main.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the RAG System

1. **Upload Documents**: Use the web interface to upload PDFs, Word documents, or images
2. **Ask Questions**: Query your documents using natural language
3. **Get Answers**: The system retrieves relevant chunks and generates answers using the local LLM
4. **Voice Input**: Use the mic recorder feature to ask questions verbally

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Size of text chunks for embedding (default: 400)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `EMBEDDING_MODEL`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `CHROMA_DB_PATH`: Location of vector database (default: "./data/chroma_db")
- `OLLAMA_MODEL`: Language model to use (default: "phi3")
- `TOP_K`: Number of documents to retrieve (default: 5)

## Supported File Formats

- **Documents**: PDF (`.pdf`), Word (`.docx`)
- **Images**: PNG, JPG, JPEG (with OCR)
- **Audio**: WAV, MP3, M4A (with Whisper)

## System Requirements by Model Size

**Phi3 (3.8B parameters):**
- 4GB RAM minimum
- Recommended for standard use cases

**Llama3.1 8B:**
- 8GB+ RAM recommended
- Better quality responses
- Slower inference

**Llama3.1 70B:**
- 32GB+ RAM required
- Recommended for high-end systems only

## Troubleshooting

**Ollama connection error:**
- Ensure Ollama is running: `ollama serve` or check system tray
- Verify connection: `curl http://localhost:11434/api/tags`

**Tesseract not found (Windows):**
- Ensure Tesseract is installed to `C:\Program Files\Tesseract-OCR`
- Update `PYTESSERACT_PATH` in `.env` if installed elsewhere

**Slow inference:**
- Model size may be too large for your system
- Try a smaller model (Phi3 vs Llama3.1)
- Add more RAM if possible

**Out of memory errors:**
- Close other applications
- Reduce `CHUNK_SIZE` in config.py
- Use a smaller model

## Development

### Running Tests

```bash
pytest tests/
```

### Project Modules

- **ingestion**: Document loading, text extraction, chunking
- **retrieval**: Vector search, ranking, context retrieval
- **app**: Streamlit interface and main application logic
- **models**: Embedding and LLM model management

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions, please check:
1. Official documentation for dependencies (LangChain, ChromaDB, Ollama)
2. Troubleshooting section above
3. Project issues/discussions
