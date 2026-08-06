"""
Vector store module for the offline multimodal RAG system.

Manages a persistent ChromaDB collection backed by sentence-transformers
embeddings. Chunking is handled upstream; this module stores one chunk at a time.
"""

import hashlib
import sys
import uuid
from pathlib import Path
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K

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

    def store_chunks(self, chunks: List[Dict]) -> int:
        """
        Embed and store a batch of chunks produced by ingestion/chunker.py.

        Each chunk must have the shape::

            {"text": str, "metadata": {"source": str, "file_type": str,
                                        "page_or_timestamp": str, ...}}

        A stable ID is derived from sha256(text + source) so exact duplicates
        (same text from the same source file) are silently skipped.

        Args:
            chunks: List of chunk dicts as returned by chunk_text().

        Returns:
            Number of chunks actually written (0 means all were duplicates).
        """
        if not chunks:
            return 0

        # Build candidate ids, de-duplicating *within this batch* first
        # (e.g. a document with repeated headers/boilerplate can otherwise
        # produce two chunks with the same text+source, which ChromaDB's
        # .get()/.add() calls reject as duplicate ids in a single call).
        candidates_by_id: Dict[str, Dict] = {}
        for chunk in chunks:
            text = chunk["text"]
            source = chunk["metadata"].get("source", "")
            chunk_id = hashlib.sha256(f"{text}{source}".encode()).hexdigest()
            candidates_by_id[chunk_id] = {"id": chunk_id, "text": text, "metadata": chunk["metadata"]}
        candidates: List[Dict] = list(candidates_by_id.values())

        # Filter out ids already present in the collection
        existing_ids = set(
            self._collection.get(ids=[c["id"] for c in candidates])["ids"]
        )
        new_chunks = [c for c in candidates if c["id"] not in existing_ids]

        if not new_chunks:
            return 0

        # Batch-embed all new chunks in one call
        texts = [c["text"] for c in new_chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        self._collection.add(
            ids=[c["id"] for c in new_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in new_chunks],
        )
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = None,
        file_type: str = None,
        source: str = None,
    ) -> List[Dict]:
        """
        Find the most semantically similar chunks to `query`.

        Args:
            query:     Natural-language query string.
            top_k:     Number of results to return. Defaults to TOP_K from config.
            file_type: If set, restrict results to chunks whose metadata
                       ``file_type`` equals this value (e.g. ``"pdf"``).
            source:    If set, restrict results to chunks from this filename
                       (e.g. ``"report.pdf"``).

        Returns:
            List of {"text": str, "metadata": dict, "score": float} sorted
            by descending relevance score (1 = perfect match, 0 = unrelated).
        """
        # Build ChromaDB where clause from whichever filters are provided
        where: Dict | None = None
        if file_type and source:
            where = {"$and": [{"file_type": {"$eq": file_type}},
                              {"source":    {"$eq": source}}]}
        elif file_type:
            where = {"file_type": {"$eq": file_type}}
        elif source:
            where = {"source": {"$eq": source}}

        # Clamp k to the number of *matching* chunks so ChromaDB doesn't error
        get_kwargs = {"where": where} if where else {}
        matching = len(self._collection.get(**get_kwargs)["ids"])
        k = min(top_k if top_k is not None else TOP_K, matching)
        if k == 0:
            return []

        query_embedding = self._embedder.encode(query).tolist()
        query_kwargs: Dict = {
            "query_embeddings": [query_embedding],
            "n_results":        k,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text":     doc,
                "metadata": meta,
                "score":    round(1 / (1 + dist), 4),
            }
            for doc, meta, dist in sorted(
                zip(docs, metadatas, distances),
                key=lambda t: t[2],   # ascending distance → best first
            )
        ]

    def reset(self) -> None:
        """Delete and recreate the collection, removing all stored chunks."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def count(self) -> int:
        """Return the total number of chunks currently stored in the collection."""
        return self._collection.count()


if __name__ == "__main__":
    store = VectorStore()

    # --- seed 4 topically distinct chunks -----------------------------------
    sample_chunks = [
        {
            "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889.",
            "metadata": {"source": "travel.txt", "file_type": "pdf", "page_or_timestamp": "page 1", "chunk_index": 0},
        },
        {
            "text": "Photosynthesis is the process by which plants convert sunlight into glucose.",
            "metadata": {"source": "biology.txt", "file_type": "pdf", "page_or_timestamp": "page 1", "chunk_index": 0},
        },
        {
            "text": "Python is a high-level programming language known for its readable syntax.",
            "metadata": {"source": "cs.txt", "file_type": "pdf", "page_or_timestamp": "page 1", "chunk_index": 0},
        },
        {
            "text": "The mitochondria is the powerhouse of the cell and produces ATP.",
            "metadata": {"source": "biology.txt", "file_type": "pdf", "page_or_timestamp": "page 2", "chunk_index": 1},
        },
    ]

    before = store.count()
    store.store_chunks(sample_chunks)
    print(f"Stored {store.count() - before} new chunk(s) ({store.count()} total)\n")

    # --- run a query and print ranked results --------------------------------
    query = "How do plants make energy from sunlight?"
    print(f"Query : {query!r}\n")

    hits = store.search(query, top_k=3)
    for rank, hit in enumerate(hits, 1):
        print(f"  #{rank}  score={hit['score']}  source={hit['metadata']['source']}")
        print(f"       {hit['text'][:100]}")
        print()
