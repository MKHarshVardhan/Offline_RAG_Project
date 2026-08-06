"""
tests/test_edge_cases.py

Pytest tests for edge-case handling across ingest.py, vector_store.py,
and rag_pipeline.py.  All fixtures are synthetic — no real media files needed.
"""

import hashlib
import io
import os
import struct
import sys
import tempfile
import wave
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make project root importable regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.ingest import ingest_file, validate_file
from config import MAX_FILE_SIZE_MB, MIN_TRANSCRIPT_WORDS


# ---------------------------------------------------------------------------
# Fixtures — synthetic file builders
# ---------------------------------------------------------------------------

def _write_tmp(content: bytes, suffix: str) -> str:
    """Write bytes to a named temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(content)
        return f.name


def _make_minimal_docx() -> bytes:
    """Return the bytes of a minimal valid (but empty-content) .docx file."""
    # A .docx is a ZIP containing at minimum [Content_Types].xml and
    # word/document.xml with an empty body.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p/></w:body></w:document>",
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>',
        )
    return buf.getvalue()


def _make_silent_wav(duration_secs: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Return bytes of a WAV file containing pure silence."""
    buf = io.BytesIO()
    n_frames = int(duration_secs * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# validate_file tests
# ---------------------------------------------------------------------------

class TestValidateFile:

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_file("/nonexistent/path/file.pdf")

    def test_empty_file_raises_value_error(self):
        path = _write_tmp(b"", ".pdf")
        try:
            with pytest.raises(ValueError, match="empty"):
                validate_file(path)
        finally:
            os.unlink(path)

    def test_oversized_file_raises_value_error(self):
        # Write a file just over the limit without allocating real memory
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024 + 1
        path = _write_tmp(b"", ".pdf")
        try:
            # Patch stat so we don't need to write gigabytes
            with patch("ingestion.ingest.Path") as mock_path_cls:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.stat.return_value = MagicMock(st_size=limit_bytes)
                mock_path.suffix = ".pdf"
                mock_path.name = "big.pdf"
                mock_path_cls.return_value = mock_path
                with pytest.raises(ValueError, match="exceeds the"):
                    validate_file(path)
        finally:
            os.unlink(path)

    def test_valid_file_passes_silently(self):
        path = _write_tmp(b"hello world", ".txt")
        try:
            validate_file(path)   # must not raise
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ingest_file — empty / no-text cases
# ---------------------------------------------------------------------------

class TestIngestFileEmpty:

    def test_empty_pdf_raises_value_error(self):
        path = _write_tmp(b"", ".pdf")
        try:
            with pytest.raises(ValueError, match="empty"):
                ingest_file(path)
        finally:
            os.unlink(path)

    def test_empty_docx_raises_value_error(self):
        path = _write_tmp(b"", ".docx")
        try:
            with pytest.raises(ValueError, match="empty"):
                ingest_file(path)
        finally:
            os.unlink(path)

    def test_empty_wav_raises_value_error(self):
        path = _write_tmp(b"", ".wav")
        try:
            with pytest.raises(ValueError, match="empty"):
                ingest_file(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ingest_file — corrupted / password-protected PDF
# ---------------------------------------------------------------------------

class TestIngestFileCorruptPDF:

    def test_corrupt_pdf_raises_value_error(self):
        # Random bytes that are not a valid PDF
        path = _write_tmp(b"NOT A PDF \x00\x01\x02\x03" * 20, ".pdf")
        try:
            with pytest.raises(ValueError, match="Could not open|password"):
                ingest_file(path)
        finally:
            os.unlink(path)

    def test_password_protected_pdf_raises_value_error(self):
        # fitz raises an error containing "password" for encrypted PDFs.
        # We mock extract_pdf to simulate that without needing a real encrypted file.
        path = _write_tmp(b"%PDF-1.4 fake", ".pdf")
        try:
            with patch("ingestion.ingest.extract_pdf",
                       side_effect=Exception("password required to open")):
                with pytest.raises(ValueError, match="password-protected"):
                    ingest_file(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ingest_file — audio silence
# ---------------------------------------------------------------------------

class TestIngestFileAudioSilence:

    def test_silent_wav_raises_value_error(self):
        path = _write_tmp(_make_silent_wav(), ".wav")
        try:
            # Mock extract_audio to return a near-empty transcript (silence)
            with patch("ingestion.ingest.extract_audio",
                       return_value={"text": "", "segments": [], "source": "silent.wav"}):
                with pytest.raises(ValueError, match="near-empty transcript|silent"):
                    ingest_file(path)
        finally:
            os.unlink(path)

    def test_short_transcript_below_min_words_raises(self):
        path = _write_tmp(_make_silent_wav(), ".wav")
        short_text = " ".join(["word"] * (MIN_TRANSCRIPT_WORDS - 1))
        try:
            with patch("ingestion.ingest.extract_audio",
                       return_value={"text": short_text, "segments": [], "source": "short.wav"}):
                with pytest.raises(ValueError, match="near-empty transcript"):
                    ingest_file(path)
        finally:
            os.unlink(path)

    def test_sufficient_transcript_does_not_raise(self):
        path = _write_tmp(_make_silent_wav(), ".wav")
        good_text = " ".join(["word"] * (MIN_TRANSCRIPT_WORDS + 5))
        segments = [{"start": 0.0, "end": 1.0, "text": good_text}]
        try:
            with patch("ingestion.ingest.extract_audio",
                       return_value={"text": good_text, "segments": segments, "source": "ok.wav"}):
                result = ingest_file(path)
                assert len(result) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ingest_file — unsupported extension
# ---------------------------------------------------------------------------

class TestIngestFileUnsupported:

    def test_unsupported_extension_raises_value_error(self):
        path = _write_tmp(b"some content", ".xyz")
        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                ingest_file(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# vector_store — duplicate chunk handling
# ---------------------------------------------------------------------------

class TestVectorStoreDuplicates:

    def _make_store(self):
        """Return an isolated in-memory VectorStore for testing."""
        import uuid
        import chromadb
        from sentence_transformers import SentenceTransformer

        store = object.__new__(__import__("retrieval.vector_store", fromlist=["VectorStore"]).VectorStore)
        store._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        store._client = chromadb.Client()
        store._collection = store._client.get_or_create_collection(f"test_{uuid.uuid4().hex}")
        return store

    def test_duplicate_chunks_not_stored_twice(self):
        store = self._make_store()
        chunk = {
            "text": "Photosynthesis converts sunlight into glucose.",
            "metadata": {"source": "bio.pdf", "file_type": "pdf",
                         "page_or_timestamp": "page 1", "chunk_index": 0},
        }
        added_first  = store.store_chunks([chunk])
        added_second = store.store_chunks([chunk])

        assert added_first  == 1
        assert added_second == 0
        assert store.count() == 1

    def test_mixed_batch_only_stores_new_chunks(self):
        store = self._make_store()
        chunk_a = {
            "text": "The mitochondria produces ATP.",
            "metadata": {"source": "bio.pdf", "file_type": "pdf",
                         "page_or_timestamp": "page 2", "chunk_index": 1},
        }
        chunk_b = {
            "text": "Python is a high-level language.",
            "metadata": {"source": "cs.pdf", "file_type": "pdf",
                         "page_or_timestamp": "page 1", "chunk_index": 0},
        }
        store.store_chunks([chunk_a])
        added = store.store_chunks([chunk_a, chunk_b])   # chunk_a is duplicate

        assert added == 1
        assert store.count() == 2

    def test_empty_chunk_list_returns_zero(self):
        store = self._make_store()
        assert store.store_chunks([]) == 0


# ---------------------------------------------------------------------------
# rag_pipeline — blank query guard
# ---------------------------------------------------------------------------

class TestRagPipelineBlankQuery:

    def test_blank_query_returns_please_enter(self):
        from app.rag_pipeline import generate_answer
        chunks = [{"text": "Some context.", "metadata": {}, "score": 0.9}]
        result = generate_answer("   ", chunks)
        assert "Please enter a question" in result["answer"]
        assert result["sources"] == []

    def test_empty_chunks_returns_no_info(self):
        from app.rag_pipeline import generate_answer, _NO_INFO
        result = generate_answer("What is photosynthesis?", [])
        assert result["answer"] == _NO_INFO
        assert result["sources"] == []
