"""
retrieval/tune_experiment.py

Runs the same query against several (chunk_size, chunk_overlap) configurations
and prints retrieved results side by side so you can pick the best defaults.

Usage:
    python retrieval/tune_experiment.py

Each configuration gets its own *isolated* in-memory ChromaDB collection so
the experiment never touches the persistent knowledge base.
"""

import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Experiment parameters — edit these to suit your corpus
# ---------------------------------------------------------------------------

QUERY = "How do plants produce energy from sunlight?"

# Representative corpus: mix of on-topic and off-topic paragraphs
CORPUS = [
    (
        "Photosynthesis is the biological process by which green plants, algae, and "
        "some bacteria convert light energy — usually from the sun — into chemical "
        "energy stored as glucose. This process takes place mainly in the chloroplasts "
        "of plant cells, using the green pigment chlorophyll to absorb sunlight. "
        "The overall reaction combines carbon dioxide from the air with water absorbed "
        "through the roots, releasing oxygen as a by-product."
    ),
    (
        "Cellular respiration is the process by which organisms break down glucose to "
        "release energy in the form of ATP. It occurs in the mitochondria and involves "
        "three main stages: glycolysis, the Krebs cycle, and the electron transport chain. "
        "Unlike photosynthesis, cellular respiration consumes oxygen and produces carbon "
        "dioxide and water."
    ),
    (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
        "France. It was constructed between 1887 and 1889 as the centerpiece of the 1889 "
        "World's Fair. Named after engineer Gustave Eiffel, whose company designed and "
        "built the tower, it stands 330 metres tall and was the world's tallest man-made "
        "structure for 41 years."
    ),
    (
        "Python is a high-level, general-purpose programming language. Its design "
        "philosophy emphasises code readability with the use of significant indentation. "
        "Python is dynamically typed and garbage-collected. It supports multiple "
        "programming paradigms, including structured, object-oriented, and functional "
        "programming. It is often described as a 'batteries included' language due to "
        "its comprehensive standard library."
    ),
    (
        "Chlorophyll is the primary pigment used in photosynthesis. It absorbs light "
        "most strongly in the blue and red portions of the electromagnetic spectrum, "
        "while reflecting green light — which is why plants appear green. There are "
        "several types of chlorophyll; chlorophyll a and chlorophyll b are the most "
        "common in land plants."
    ),
]

# Configurations to compare: (chunk_size_words, chunk_overlap_words)
CONFIGS: List[Tuple[int, int]] = [
    (50,  10),
    (100, 20),
    (200, 40),
    (400, 50),
]

TOP_K   = 3
COL_W   = 52   # display width per column
SNIPPET = 200  # max chars shown per result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    step  = max(1, size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks


def _build_collection(size: int, overlap: int, embedder: SentenceTransformer):
    """Create an isolated in-memory collection for one configuration."""
    client     = chromadb.Client()          # ephemeral, in-memory
    collection = client.get_or_create_collection(f"exp_{size}_{overlap}")

    all_texts: List[str] = []
    for para in CORPUS:
        all_texts.extend(_chunk(para, size, overlap))

    if not all_texts:
        return collection

    embeddings = embedder.encode(all_texts, show_progress_bar=False).tolist()
    ids        = [str(i) for i in range(len(all_texts))]
    collection.add(ids=ids, embeddings=embeddings, documents=all_texts)
    return collection


def _search(collection, query_embedding: List[float], k: int) -> List[dict]:
    k = min(k, collection.count())
    if k == 0:
        return []
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"],
    )
    return [
        {"text": doc, "score": round(1 / (1 + dist), 4)}
        for doc, dist in zip(res["documents"][0], res["distances"][0])
    ]


def _wrap(text: str, width: int) -> List[str]:
    return textwrap.wrap(text[:SNIPPET], width) or ["(empty)"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    print(f"\nQuery: {QUERY!r}\n")
    print(f"Corpus: {len(CORPUS)} paragraphs  |  top_k={TOP_K}\n")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = embedder.encode(QUERY).tolist()

    # Collect results for every config
    all_results: List[Tuple[Tuple[int, int], List[dict], int]] = []
    for size, overlap in CONFIGS:
        col = _build_collection(size, overlap, embedder)
        hits = _search(col, query_embedding, TOP_K)
        n_chunks = col.count()
        all_results.append(((size, overlap), hits, n_chunks))

    # ── Header row ──────────────────────────────────────────────────────────
    header_parts = []
    for (size, overlap), _, n_chunks in all_results:
        label = f"size={size} overlap={overlap} ({n_chunks} chunks)"
        header_parts.append(label.center(COL_W))
    print("  ".join(header_parts))
    print("  ".join(["─" * COL_W] * len(all_results)))

    # ── Results rows ────────────────────────────────────────────────────────
    for rank in range(TOP_K):
        # Collect wrapped lines for each config at this rank
        col_lines: List[List[str]] = []
        for (size, overlap), hits, _ in all_results:
            if rank < len(hits):
                hit   = hits[rank]
                score = f"#{rank+1} score={hit['score']}"
                lines = [score] + _wrap(hit["text"], COL_W)
            else:
                lines = ["(no result)"]
            col_lines.append(lines)

        # Print side by side, padding shorter columns with blank lines
        max_lines = max(len(c) for c in col_lines)
        for col in col_lines:
            col += [""] * (max_lines - len(col))

        for row in range(max_lines):
            print("  ".join(line.ljust(COL_W) for line in [c[row] for c in col_lines]))
        print("  ".join(["·" * COL_W] * len(all_results)))

    # ── Summary table ───────────────────────────────────────────────────────
    print("\nSummary — top-1 score per configuration:")
    print(f"  {'Config':<25} {'Chunks':>7}  {'Top-1 score':>11}  Top-1 snippet")
    print(f"  {'─'*25}  {'─'*6}  {'─'*11}  {'─'*30}")
    for (size, overlap), hits, n_chunks in all_results:
        cfg     = f"size={size}, overlap={overlap}"
        score   = hits[0]["score"] if hits else 0.0
        snippet = hits[0]["text"][:50].replace("\n", " ") + "…" if hits else "—"
        print(f"  {cfg:<25}  {n_chunks:>6}  {score:>11.4f}  {snippet}")

    print(
        "\nTip: higher top-1 score = query is closer to the best chunk. "
        "Prefer configs where the top-1 score is clearly higher than #2/#3."
    )


if __name__ == "__main__":
    run_experiment()
