"""
Vector store module for the offline multimodal RAG system.

Manages a persistent ChromaDB collection backed by sentence-transformers
embeddings. Chunking is handled upstream; this module stores one chunk at a time.
"""

import sys
import uuid
from pathlib import Path
from typing import Dict

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHROMA_DB_PATH, EMBEDDING_MODEL

COLLECTION_NAME = "documents"


class VectorStore:
    """Persistent ChromaDB-backed vector store with local sentence-transformer embeddings."""

    def __init__(self):
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_document(self, text: str, metadata: Dict) -> None:
        """
        Embed and store a single text chunk with its metadata.

        Args:
            text:     The text chunk to embed and store.
            metadata: Arbitrary key-value metadata (e.g. source, page_number).
        """
        embedding = self._embedder.encode(text).tolist()
        self._collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def count(self) -> int:
        """Return the total number of chunks currently stored in the collection."""
        return self._collection.count()


if __name__ == "__main__":
    store = VectorStore()

    before = store.count()
    print(f"Chunks before : {before}")

    store.add_document(
        text="The Eiffel Tower is located in Paris, France.",
        metadata={"source": "sample.txt", "page_number": 1},
    )

    after = store.count()
    print(f"Chunks after  : {after}")
    print(f"Stored        : {'✓ confirmed' if after == before + 1 else '✗ failed'}")
