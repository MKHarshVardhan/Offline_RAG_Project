"""
ingestion/ingest.py

Unified entry point for all file ingestion.
Detects file type, routes to the correct extractor, applies OCR fallback
for scanned PDF pages, normalises output, and cleans extracted text.
"""

import re
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.pdf_extractor import extract_pdf
from ingestion.docx_extractor import extract_docx
from ingestion.image_extractor import extract_image
from ingestion.audio_extractor import extract_audio
from config import MAX_FILE_SIZE_MB, MIN_TRANSCRIPT_WORDS

# How many words to accumulate before starting a new DOCX block. Kept smaller
# than CHUNK_SIZE (400) since chunk_text() still runs on top of this and can
# add its own overlap; this just ensures related short paragraphs/table rows
# aren't split into disconnected single-line chunks.
DOCX_GROUP_WORDS = 150
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Common header/footer patterns (page numbers, "Confidential", running titles)
_HEADER_FOOTER_RE = re.compile(
    r"(?m)^\s*(?:page\s+\d+\s*(?:of\s+\d+)?|confidential|draft|\d+)\s*$",
    re.IGNORECASE,
)
# Non-printable / control characters (keep normal whitespace)
_JUNK_CHARS_RE = re.compile(r"[^\x20-\x7E\n\t]")


def _clean(text: str) -> str:
    """Remove junk chars, strip header/footer lines, collapse whitespace."""
    text = _JUNK_CHARS_RE.sub(" ", text)
    text = _HEADER_FOOTER_RE.sub("", text)
    # Collapse runs of whitespace / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """Render a single PDF page to a temp PNG and run OCR on it."""
    import fitz  # PyMuPDF – already required by pdf_extractor

    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]          # page_number is 1-indexed
    mat = fitz.Matrix(2.0, 2.0)          # 2× zoom → better OCR accuracy
    pix = page.get_pixmap(matrix=mat)
    doc.close()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pix.save(tmp_path)
        result = extract_image(tmp_path)
        return result["text"]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_file(file_path: str) -> None:
    """
    Run pre-ingestion checks and raise a descriptive error on failure.

    Checks (in order):
    1. File exists.
    2. File is not 0 bytes.
    3. File size does not exceed MAX_FILE_SIZE_MB.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError:        File is empty or exceeds the size limit.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"'{path.name}' is empty (0 bytes).")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"'{path.name}' is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit. Split the file or raise "
            "MAX_FILE_SIZE_MB in config.py."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(file_path: str) -> list[dict]:
    """
    Ingest a file and return normalised, cleaned chunks ready for embedding.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        List of {"text": str, "metadata": {"source": str,
                                            "file_type": str,
                                            "page_or_timestamp": str}}

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file extension is not supported.
    """
    path = Path(file_path)
    validate_file(file_path)   # raises FileNotFoundError / ValueError early

    ext = path.suffix.lower()
    source = path.name
    results: list[dict] = []

    # ------------------------------------------------------------------ PDF
    if ext == ".pdf":
        try:
            pages = extract_pdf(file_path)
        except Exception as exc:
            msg = str(exc).lower()
            if "password" in msg or "encrypted" in msg:
                raise ValueError(
                    f"'{source}' is password-protected. "
                    "Please provide an unlocked copy."
                ) from exc
            raise ValueError(f"Could not open '{source}' as a PDF: {exc}") from exc

        for page in pages:
            text = page["text"]
            if page["needs_ocr"] or not text.strip():
                text = _ocr_pdf_page(file_path, page["page_number"])
            results.append({
                "text": _clean(text),
                "metadata": {
                    "source": source,
                    "file_type": "pdf",
                    "page_or_timestamp": f"page {page['page_number']}",
                },
            })

    # ----------------------------------------------------------------- DOCX
    elif ext == ".docx":
        # Group consecutive short paragraphs/table-rows into merged blocks of
        # roughly DOCX_GROUP_WORDS words each, instead of treating every single
        # paragraph as its own isolated chunk. Otherwise closely-related lines
        # (e.g. a "Team 18" label immediately followed by its "Project Title: ..."
        # line) end up in two disconnected chunks that retrieval can't link back
        # together.
        buffer_texts: list[str] = []
        buffer_word_count = 0
        first_idx = last_idx = None

        def _flush():
            if not buffer_texts:
                return
            label = (f"block {first_idx}" if first_idx == last_idx
                     else f"blocks {first_idx}-{last_idx}")
            results.append({
                "text": _clean("\n".join(buffer_texts)),
                "metadata": {
                    "source": source,
                    "file_type": "docx",
                    "page_or_timestamp": label,
                },
            })

        for block in extract_docx(file_path):
            text = block["text"]
            if first_idx is None:
                first_idx = block["paragraph_index"]
            last_idx = block["paragraph_index"]
            buffer_texts.append(text)
            buffer_word_count += len(text.split())

            if buffer_word_count >= DOCX_GROUP_WORDS:
                _flush()
                buffer_texts, buffer_word_count = [], 0
                first_idx = None

        _flush()  # emit whatever's left over

    # ---------------------------------------------------------------- Image
    elif ext in {".png", ".jpg", ".jpeg"}:
        raw = extract_image(file_path)
        results.append({
            "text": _clean(raw["text"]),
            "metadata": {
                "source": source,
                "file_type": "image",
                "page_or_timestamp": "page 1",
            },
        })

    # ---------------------------------------------------------------- Audio
    elif ext in {".wav", ".mp3", ".m4a"}:
        raw = extract_audio(file_path)
        word_count = len(raw["text"].split())
        if word_count < MIN_TRANSCRIPT_WORDS:
            raise ValueError(
                f"'{source}' produced a near-empty transcript ({word_count} word(s)). "
                "The recording may be silent or contain only background noise."
            )
        # Emit one entry per segment so timestamps are preserved
        for seg in raw["segments"]:
            results.append({
                "text": _clean(seg["text"]),
                "metadata": {
                    "source": source,
                    "file_type": "audio",
                    "page_or_timestamp": f"{seg['start']}s-{seg['end']}s",
                },
            })
        # Fallback: no segments → single entry with full transcript
        if not raw["segments"] and raw["text"]:
            results.append({
                "text": _clean(raw["text"]),
                "metadata": {
                    "source": source,
                    "file_type": "audio",
                    "page_or_timestamp": "0s",
                },
            })

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Supported: .pdf, .docx, .png, .jpg, .jpeg, .wav, .mp3, .m4a"
        )

    # Drop entries where cleaning left nothing
    return [r for r in results if r["text"]]


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion/ingest.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"Ingesting: {file_path}\n")

    try:
        chunks = ingest_file(file_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Total entries produced: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]):          # preview first 5
        meta = chunk["metadata"]
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"[{i}] {meta['file_type']} | {meta['page_or_timestamp']}")
        print(f"     {preview!r}")
        print()

    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more entries.")
