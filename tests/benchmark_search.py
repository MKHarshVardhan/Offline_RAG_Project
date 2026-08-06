"""
tests/benchmark_search.py

Times VectorStore.search() across 20 sample queries and reports:
  - per-query latency
  - average, median, p95, min, max
  - breakdown of time spent in embedding vs ChromaDB query

Safe to run at any time: uses an isolated in-memory collection seeded with
a representative corpus, never touches the production DB.

Usage:
    python tests/benchmark_search.py

    # Run with more iterations for a tighter estimate:
    python tests/benchmark_search.py --runs 50
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TOP_K
from retrieval.vector_store import COLLECTION_NAME

# ---------------------------------------------------------------------------
# Corpus — 20 topically varied paragraphs
# ---------------------------------------------------------------------------

CORPUS = [
    ("Photosynthesis is the process by which plants convert sunlight into glucose "
     "using chlorophyll in the chloroplasts.", "biology.pdf", "page 1"),
    ("The mitochondria is the powerhouse of the cell, producing ATP via the "
     "electron transport chain.", "biology.pdf", "page 2"),
    ("DNA replication occurs during the S phase of the cell cycle and is "
     "catalysed by DNA polymerase.", "biology.pdf", "page 3"),
    ("The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889 "
     "by Gustave Eiffel.", "travel.pdf", "page 1"),
    ("The Great Wall of China stretches over 21,000 kilometres and was built "
     "over many centuries.", "travel.pdf", "page 2"),
    ("Python is a high-level programming language known for its readable syntax "
     "and large standard library.", "cs.pdf", "page 1"),
    ("Machine learning is a subset of artificial intelligence that enables "
     "systems to learn from data.", "cs.pdf", "page 2"),
    ("A neural network consists of layers of interconnected nodes that transform "
     "input data into predictions.", "cs.pdf", "page 3"),
    ("The water cycle describes the continuous movement of water through "
     "evaporation, condensation, and precipitation.", "geography.pdf", "page 1"),
    ("Plate tectonics explains the movement of Earth's lithospheric plates and "
     "the formation of mountains and earthquakes.", "geography.pdf", "page 2"),
    ("The French Revolution began in 1789 and led to the abolition of the "
     "French monarchy.", "history.pdf", "page 1"),
    ("World War II ended in 1945 with the surrender of Germany and Japan.", "history.pdf", "page 2"),
    ("Supply and demand is a fundamental economic model describing price "
     "determination in a market.", "economics.pdf", "page 1"),
    ("Inflation refers to the general increase in prices and the corresponding "
     "decrease in purchasing power.", "economics.pdf", "page 2"),
    ("Shakespeare wrote 37 plays and 154 sonnets during the Elizabethan era.", "literature.pdf", "page 1"),
    ("The speed of light in a vacuum is approximately 299,792 kilometres per second.", "physics.pdf", "page 1"),
    ("Newton's second law states that force equals mass multiplied by acceleration.", "physics.pdf", "page 2"),
    ("The periodic table organises chemical elements by atomic number and "
     "chemical properties.", "chemistry.pdf", "page 1"),
    ("Antibiotics are medicines that kill or inhibit the growth of bacteria "
     "and are ineffective against viruses.", "medicine.pdf", "page 1"),
    ("The human genome contains approximately 3 billion base pairs encoding "
     "around 20,000 protein-coding genes.", "medicine.pdf", "page 2"),
]

# ---------------------------------------------------------------------------
# 20 benchmark queries — deliberately varied in topic and phrasing
# ---------------------------------------------------------------------------

QUERIES: List[str] = [
    "How do plants make energy from sunlight?",
    "What is the role of mitochondria in cells?",
    "Explain DNA replication.",
    "Where is the Eiffel Tower located?",
    "How long is the Great Wall of China?",
    "What makes Python a popular programming language?",
    "What is machine learning?",
    "How does a neural network work?",
    "Describe the water cycle.",
    "What causes earthquakes?",
    "When did the French Revolution start?",
    "How did World War II end?",
    "What is supply and demand?",
    "What does inflation mean?",
    "How many plays did Shakespeare write?",
    "What is the speed of light?",
    "State Newton's second law.",
    "What is the periodic table?",
    "What are antibiotics used for?",
    "How many genes does the human genome contain?",
]

assert len(QUERIES) == 20, "Benchmark requires exactly 20 queries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: List[float], pct: float) -> float:
    idx = int(len(sorted_values) * pct / 100)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def _build_store(embedder: SentenceTransformer):
    """Seed an in-memory collection with the corpus and return it."""
    client     = chromadb.Client()   # ephemeral, never touches disk
    collection = client.get_or_create_collection(COLLECTION_NAME)

    texts      = [c[0] for c in CORPUS]
    metadatas  = [{"source": c[1], "file_type": "pdf",
                   "page_or_timestamp": c[2], "chunk_index": i}
                  for i, c in enumerate(CORPUS)]
    ids        = [str(i) for i in range(len(CORPUS))]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(ids=ids, embeddings=embeddings,
                   documents=texts, metadatas=metadatas)
    return collection


def _search_timed(collection, embedder: SentenceTransformer,
                  query: str, top_k: int) -> tuple[list, float, float]:
    """
    Run one search and return (hits, embed_ms, query_ms).
    Splits timing between the embedding step and the ChromaDB query step.
    """
    t0 = time.perf_counter()
    query_embedding = embedder.encode(query).tolist()
    t1 = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    t2 = time.perf_counter()

    hits = [
        {"text": doc, "metadata": meta, "score": round(1 / (1 + dist), 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    return hits, (t1 - t0) * 1000, (t2 - t1) * 1000


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_runs: int = 1) -> None:
    print(f"\n{'='*64}")
    print(f"  VectorStore.search() Benchmark")
    print(f"  Corpus : {len(CORPUS)} chunks  |  Queries : {len(QUERIES)}  |  Runs : {n_runs}")
    print(f"  Model  : {EMBEDDING_MODEL}  |  top_k : {TOP_K}")
    print(f"{'='*64}\n")

    print("Loading embedding model and seeding corpus…", end=" ", flush=True)
    embedder  = SentenceTransformer(EMBEDDING_MODEL)
    collection = _build_store(embedder)
    print("done.\n")

    # Warm-up: one pass through all queries (not recorded)
    print("Warming up (1 pass, not recorded)…", end=" ", flush=True)
    for q in QUERIES:
        _search_timed(collection, embedder, q, TOP_K)
    print("done.\n")

    # Timed runs
    total_ms:  List[float] = []
    embed_ms:  List[float] = []
    chroma_ms: List[float] = []
    per_query: dict[str, List[float]] = {q: [] for q in QUERIES}

    print(f"{'Query':<52} {'Avg ms':>8}  {'Top-1 source'}")
    print(f"{'─'*52}  {'─'*8}  {'─'*20}")

    for q in QUERIES:
        run_times: List[float] = []
        last_hits = []
        for _ in range(n_runs):
            hits, em, cm = _search_timed(collection, embedder, q, TOP_K)
            elapsed = em + cm
            run_times.append(elapsed)
            embed_ms.append(em)
            chroma_ms.append(cm)
            total_ms.append(elapsed)
            last_hits = hits

        avg = sum(run_times) / len(run_times)
        per_query[q] = run_times
        top_src = last_hits[0]["metadata"]["source"] if last_hits else "—"
        print(f"  {q[:50]:<50}  {avg:>7.1f}ms  {top_src}")

    # ── Aggregate statistics ─────────────────────────────────────────────
    total_ms.sort()
    embed_ms.sort()
    chroma_ms.sort()

    n = len(total_ms)
    avg_total  = sum(total_ms)  / n
    avg_embed  = sum(embed_ms)  / n
    avg_chroma = sum(chroma_ms) / n

    print(f"\n{'─'*64}")
    print(f"  {'Metric':<30} {'Total':>10}  {'Embed':>10}  {'ChromaDB':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*10}")

    rows = [
        ("Average",  avg_total,                        avg_embed,                        avg_chroma),
        ("Median",   _percentile(total_ms,  50),        _percentile(embed_ms,  50),        _percentile(chroma_ms,  50)),
        ("p95",      _percentile(total_ms,  95),        _percentile(embed_ms,  95),        _percentile(chroma_ms,  95)),
        ("Min",      total_ms[0],                       embed_ms[0],                       chroma_ms[0]),
        ("Max",      total_ms[-1],                      embed_ms[-1],                      chroma_ms[-1]),
    ]
    for label, tot, emb, chm in rows:
        print(f"  {label:<30}  {tot:>9.2f}ms  {emb:>9.2f}ms  {chm:>9.2f}ms")

    print(f"\n  Embed share  : {avg_embed  / avg_total * 100:.1f}% of total latency")
    print(f"  ChromaDB share: {avg_chroma / avg_total * 100:.1f}% of total latency")
    print(f"\n  Total queries timed : {n}")
    print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VectorStore.search() latency.")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of timed passes per query (default: 1). Use 3–5 for stable averages.",
    )
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs)
