"""
tests/check_persistence.py

Proves that ChromaDB data written by one VectorStore instance is fully
retrievable by a brand-new instance pointing at the same path — i.e. that
nothing is held only in memory and lost when the client is closed.

Safe to run at any time: uses an isolated temp directory, never touches the
production DB at CHROMA_DB_PATH.

Usage:
    python tests/check_persistence.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
from retrieval.vector_store import COLLECTION_NAME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "text": "Photosynthesis converts sunlight into glucose using chlorophyll.",
        "metadata": {"source": "bio.pdf", "file_type": "pdf",
                     "page_or_timestamp": "page 1", "chunk_index": 0},
    },
    {
        "text": "The mitochondria produces ATP through cellular respiration.",
        "metadata": {"source": "bio.pdf", "file_type": "pdf",
                     "page_or_timestamp": "page 2", "chunk_index": 1},
    },
    {
        "text": "The Eiffel Tower was built in Paris in 1889.",
        "metadata": {"source": "travel.pdf", "file_type": "pdf",
                     "page_or_timestamp": "page 1", "chunk_index": 0},
    },
]

QUERY = "How do plants produce energy?"


def _make_store(db_path: str):
    """
    Return a minimal VectorStore-like object backed by a PersistentClient at
    db_path.  We build it directly (not via VectorStore.__init__) so we can
    control the path without monkey-patching config.
    """
    from retrieval.vector_store import VectorStore
    store = object.__new__(VectorStore)
    store._embedder   = SentenceTransformer(EMBEDDING_MODEL)
    store._client     = chromadb.PersistentClient(path=db_path)
    store._collection = store._client.get_or_create_collection(COLLECTION_NAME)
    return store


def _pass(msg: str) -> None:
    print(f"  ✅  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌  FAIL  {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Persistence checks
# ---------------------------------------------------------------------------

def check_persistence() -> None:
    db_dir = tempfile.mkdtemp(prefix="rag_persist_test_")
    print(f"\nUsing temp DB: {db_dir}\n")

    try:
        # ── Phase 1: write ──────────────────────────────────────────────────
        print("Phase 1 — Writing chunks with instance A …")
        store_a = _make_store(db_dir)
        added   = store_a.store_chunks(SAMPLE_CHUNKS)

        if added == len(SAMPLE_CHUNKS):
            _pass(f"store_chunks() wrote {added} chunk(s)")
        else:
            _fail(f"Expected {len(SAMPLE_CHUNKS)} chunks written, got {added}")

        count_a = store_a.count()
        if count_a == len(SAMPLE_CHUNKS):
            _pass(f"count() == {count_a} immediately after write")
        else:
            _fail(f"count() returned {count_a}, expected {len(SAMPLE_CHUNKS)}")

        # Explicitly delete the Python object — this closes the underlying
        # SQLite connection that ChromaDB's PersistentClient holds.
        del store_a
        print()

        # ── Phase 2: read from a fresh instance ─────────────────────────────
        print("Phase 2 — Reading back with a brand-new instance B …")
        store_b = _make_store(db_dir)

        count_b = store_b.count()
        if count_b == len(SAMPLE_CHUNKS):
            _pass(f"count() == {count_b} after client reopen (data survived)")
        else:
            _fail(f"count() returned {count_b} after reopen, expected {len(SAMPLE_CHUNKS)}")

        # ── Phase 3: verify search returns correct content ──────────────────
        print()
        print(f"Phase 3 — Searching with query: {QUERY!r} …")
        hits = store_b.search(QUERY, top_k=2)

        if len(hits) == 2:
            _pass(f"search() returned {len(hits)} result(s)")
        else:
            _fail(f"search() returned {len(hits)} result(s), expected 2")

        top_text   = hits[0]["text"]
        top_source = hits[0]["metadata"]["source"]
        top_score  = hits[0]["score"]

        # The biology chunk should rank first for this query
        if "photosynthesis" in top_text.lower() or "mitochondria" in top_text.lower():
            _pass(f"Top result is biology-related (score={top_score}): {top_text[:60]!r}")
        else:
            _fail(f"Unexpected top result: {top_text[:80]!r}")

        # Verify all metadata fields survived the round-trip
        for hit in hits:
            meta = hit["metadata"]
            missing = [k for k in ("source", "file_type", "page_or_timestamp") if k not in meta]
            if missing:
                _fail(f"Metadata fields missing after reopen: {missing}")
        _pass("All metadata fields intact after reopen")

        # ── Phase 4: duplicate re-write returns 0 ───────────────────────────
        print()
        print("Phase 4 — Re-writing same chunks (should all be skipped) …")
        re_added = store_b.store_chunks(SAMPLE_CHUNKS)
        if re_added == 0:
            _pass(f"store_chunks() correctly skipped {len(SAMPLE_CHUNKS)} duplicate(s)")
        else:
            _fail(f"Expected 0 new chunks on re-write, got {re_added}")

        final_count = store_b.count()
        if final_count == len(SAMPLE_CHUNKS):
            _pass(f"Total count unchanged at {final_count} after duplicate write")
        else:
            _fail(f"Count changed to {final_count} after duplicate write")

        print("\n✅  All persistence checks passed.\n")

    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_persistence()
