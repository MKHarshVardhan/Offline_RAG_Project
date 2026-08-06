"""
ingestion/chunker.py

Splits text into overlapping word-based chunks and attaches metadata to each.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, metadata: dict) -> list[dict]:
    """
    Split `text` into overlapping chunks (measured in words).

    Args:
        text:     The full text to chunk.
        metadata: Arbitrary metadata (e.g. source filename, page number)
                  attached to every produced chunk.

    Returns:
        List of {"text": <chunk_text>, "metadata": <metadata + chunk_index>}
    """
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []

    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + CHUNK_SIZE]
        if not chunk_words:
            break
        chunks.append({
            "text": " ".join(chunk_words),
            "metadata": {**metadata, "chunk_index": i},
        })
        # Stop after the last window that starts within the text
        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


if __name__ == "__main__":
    sample = (
        "The quick brown fox jumps over the lazy dog. " * 60  # ~480 words
    )

    meta = {"source": "sample.txt", "page": 1}
    results = chunk_text(sample, meta)

    print(f"Total chunks: {len(results)}\n")
    for chunk in results:
        word_count = len(chunk["text"].split())
        print(f"  chunk_index={chunk['metadata']['chunk_index']}  words={word_count}")
        print(f"  first 12 words : {' '.join(chunk['text'].split()[:12])}")
        print(f"  last  12 words : {' '.join(chunk['text'].split()[-12:])}")
        print()
