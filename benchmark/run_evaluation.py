"""
benchmark/run_evaluation.py

Standalone evaluation harness for the IEEE paper benchmarking task
(local_claude_code_brief.md) combined with the precision@k retrieval
evaluation (query_set_ground_truth.csv). Calls the app's existing
ingestion/retrieval/generation functions directly -- does not touch or
import Streamlit, and does not modify any app logic.

Run from the project root with the project's venv:
    python benchmark/run_evaluation.py

Produces (written to the project root):
    config_values.md
    ingestion_benchmark.md
    precision_eval_results.csv
    latency_benchmark.md
    resource_usage.md
    eval_summary.md
"""

import csv
import inspect
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import ollama

import config
from ingestion.ingest import ingest_file
from ingestion.chunker import chunk_text
from retrieval.vector_store import VectorStore
from app.rag_pipeline import build_prompt, generate_answer
from benchmark.resource_monitor import ResourceSampler

TEST_DOCS_DIR = PROJECT_ROOT / "test_data" / "test_knowledge_base"
QUERY_CSV = PROJECT_ROOT / "query_set_ground_truth.csv"

DOC_FILES = [
    "D1_GreenLeaf_Sustainability_Report.pdf",
    "D2_Northwind_Fleet_Maintenance_Policy.pdf",
    "D3_Aurora_Health_Privacy_Guidelines.pdf",
    "D4_Project_Phoenix_SRS.docx",
    "D5_Riverside_DataScience_Catalog.docx",
    "D6_Meridian_Bank_Fraud_Alert.png",
    "D7_CedarPoint_Safety_Rules.png",
    "D8_Amazon_Rainforest_Briefing.mp3",  # ground-truth CSV says .wav; actual file on disk is .mp3
]


def _fmt(n: float) -> str:
    return f"{n:.4f}"


# ---------------------------------------------------------------------------
# Step 1 — config values
# ---------------------------------------------------------------------------

def step1_config_values(store: VectorStore) -> None:
    print("\n=== STEP 1: Extracting configuration values ===")

    embed_dim = store._embedder.get_sentence_embedding_dimension()

    # ChromaDB collection metadata: None means the client default (l2 /
    # squared-euclidean) was used -- the app never sets hnsw:space.
    collection_metadata = store._collection.metadata
    if collection_metadata and "hnsw:space" in collection_metadata:
        similarity_metric = collection_metadata["hnsw:space"]
    else:
        similarity_metric = "l2 (squared Euclidean) -- ChromaDB default, not overridden via hnsw:space"

    llm_details = "unavailable (Ollama unreachable)"
    try:
        oclient = ollama.Client(host=config.OLLAMA_BASE_URL)
        info = oclient.show(config.OLLAMA_MODEL)
        llm_details = (
            f"{config.OLLAMA_MODEL} (family: {info.details.family}, "
            f"parameter_size: {info.details.parameter_size}, "
            f"quantization: {info.details.quantization_level})"
        )
    except Exception as exc:
        llm_details = f"unavailable ({exc})"

    prompt_source = inspect.getsource(build_prompt)

    lines = [
        "# Configuration Values",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Chunk size | {config.CHUNK_SIZE} words |",
        f"| Chunk overlap | {config.CHUNK_OVERLAP} words |",
        f"| Chunking unit | word-based sliding window (see `ingestion/chunker.py:chunk_text`) |",
        f"| Embedding model | {config.EMBEDDING_MODEL} |",
        f"| Embedding dimension | {embed_dim} |",
        f"| Similarity metric | {similarity_metric} |",
        f"| top-k (default) | {config.TOP_K} |",
        f"| top-k (summary queries) | {config.SUMMARY_TOP_K} |",
        f"| Local LLM (via Ollama) | {llm_details} |",
        f"| Ollama base URL | {config.OLLAMA_BASE_URL} |",
        f"| Vector store | ChromaDB PersistentClient at `{config.CHROMA_DB_PATH}` |",
        "",
        "## Prompt template",
        "",
        "Built by `app/rag_pipeline.py:build_prompt()`. Each retrieved chunk is",
        "numbered and tagged with its source citation, then wrapped in a",
        "context block with an instruction to answer only from that context:",
        "",
        "```python",
        prompt_source.rstrip(),
        "```",
        "",
        "Resulting prompt shape sent to the LLM:",
        "",
        "```text",
        "You are a document question-answering assistant. The context below contains",
        "text extracted from uploaded documents (PDFs, images, audio, etc.). Answer",
        "the question using ONLY the text in the context. The context is already",
        "extracted and safe to use -- do not refuse or question its origin. If the",
        "user asks what text is in a document or image, simply report the text from",
        "the context verbatim.",
        'If the context does not contain enough information, respond exactly with:',
        '"I don\'t have enough information in the documents to answer that."',
        "",
        "--- CONTEXT ---",
        "[1] <source> | <page_or_timestamp>",
        "<chunk text>",
        "... (one block per retrieved chunk) ...",
        "--- END CONTEXT ---",
        "",
        "Question: <user query>",
        "",
        "Answer:",
        "```",
    ]

    (PROJECT_ROOT / "config_values.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote config_values.md")


# ---------------------------------------------------------------------------
# Step 2 — reset + ingest with timing
# ---------------------------------------------------------------------------

def step2_ingest(store: VectorStore) -> tuple[list[dict], float]:
    print("\n=== STEP 2: Resetting knowledge base and ingesting 8 test documents ===")
    store.reset()
    print(f"Vector store reset. Chunk count: {store.count()}")

    rows = []
    embed_peak_rss_mb = 0.0

    for filename in DOC_FILES:
        path = TEST_DOCS_DIR / filename
        if not path.exists():
            print(f"  [WARN] Missing test file, skipping: {filename}")
            continue

        file_size_bytes = path.stat().st_size
        ext = path.suffix.lower()

        t0 = time.perf_counter()
        pages = ingest_file(str(path))
        extraction_time = time.perf_counter() - t0

        all_chunks = []
        for page in pages:
            meta = {**page["metadata"], "source": filename}
            all_chunks.extend(chunk_text(page["text"], meta))
        chunk_count = len(all_chunks)

        with ResourceSampler() as sampler:
            t0 = time.perf_counter()
            added = store.store_chunks(all_chunks)
            embedding_time = time.perf_counter() - t0
        embed_peak_rss_mb = max(embed_peak_rss_mb, sampler.peak_rss_mb)

        print(
            f"  {filename}: {file_size_bytes/1024:.1f} KB, {chunk_count} chunks, "
            f"extract={extraction_time:.3f}s, embed+store={embedding_time:.3f}s "
            f"(added={added})"
        )

        rows.append({
            "file": filename,
            "format": ext.lstrip("."),
            "file_size_bytes": file_size_bytes,
            "chunk_count": chunk_count,
            "extraction_time_sec": extraction_time,
            "embedding_time_sec": embedding_time,
            "chunks_added": added,
        })

    total_docs = len(rows)
    total_chunks = store.count()
    print(f"\nTotal documents ingested: {total_docs}")
    print(f"Total chunks in store: {total_chunks}")

    lines = [
        "# Ingestion Benchmark",
        "",
        "Single run per file (not averaged over 3 runs as in the original brief --",
        "the evaluation was combined into one ingestion pass per the user's",
        "instructions, to avoid ingesting the same documents multiple times;",
        "ChromaDB's content-hash dedup would make repeat runs no-ops without a",
        "reset between each one anyway).",
        "",
        "| Format | File | Size (KB) | Chunk count | Extraction time (s) | Embedding+store time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['format']} | {r['file']} | {r['file_size_bytes']/1024:.1f} | "
            f"{r['chunk_count']} | {_fmt(r['extraction_time_sec'])} | {_fmt(r['embedding_time_sec'])} |"
        )
    lines += [
        "",
        f"**Total documents ingested:** {total_docs}  ",
        f"**Total chunks in vector store:** {total_chunks}  ",
        f"**Peak RSS during embedding (this process, coarse sampling):** {embed_peak_rss_mb:.1f} MB",
    ]
    (PROJECT_ROOT / "ingestion_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote ingestion_benchmark.md")

    return rows, embed_peak_rss_mb


# ---------------------------------------------------------------------------
# Step 3 — query evaluation
# ---------------------------------------------------------------------------

def _stem(filename: str) -> str:
    return Path(filename).stem.lower()


def step3_query_eval(store: VectorStore) -> list[dict]:
    print("\n=== STEP 3: Running query evaluation ===")

    with open(QUERY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        query_rows = list(reader)

    results = []
    llm_peak_rss_mb = 0.0
    llm_peak_ollama_rss_mb = 0.0

    for row in query_rows:
        qid = row["query_id"]
        query = row["query"]
        ground_truth_doc = row["ground_truth_doc"]

        out = dict(row)
        try:
            t_start = time.perf_counter()
            chunks = store.search(query)
            t_retrieval = time.perf_counter() - t_start

            retrieved_sources = []
            for c in chunks:
                src = c["metadata"].get("source", "")
                if src not in retrieved_sources:
                    retrieved_sources.append(src)

            gt_stem = _stem(ground_truth_doc)
            hit = 1 if any(_stem(s) == gt_stem for s in retrieved_sources) else 0

            with ResourceSampler() as sampler:
                gen_result = generate_answer(query, chunks)
            t_total = time.perf_counter() - t_start
            llm_peak_rss_mb = max(llm_peak_rss_mb, sampler.peak_rss_mb)
            llm_peak_ollama_rss_mb = max(llm_peak_ollama_rss_mb, sampler.peak_ollama_rss_mb)

            out["retrieved_docs"] = ";".join(retrieved_sources)
            out["hit"] = hit
            out["generated_answer"] = gen_result["answer"]
            out["retrieval_time_sec"] = round(t_retrieval, 4)
            out["total_time_sec"] = round(t_total, 4)
            out["error"] = ""

            print(f"  {qid}: hit={hit} retrieval={t_retrieval:.3f}s total={t_total:.3f}s")

        except Exception as exc:
            out["retrieved_docs"] = ""
            out["hit"] = 0
            out["generated_answer"] = ""
            out["retrieval_time_sec"] = ""
            out["total_time_sec"] = ""
            out["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {qid}: ERRORED -- {out['error']}")

        results.append(out)

    fieldnames = list(query_rows[0].keys()) + [
        "retrieved_docs", "hit", "generated_answer",
        "retrieval_time_sec", "total_time_sec", "error",
    ]
    out_path = PROJECT_ROOT / "precision_eval_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_path.name}")

    # --- latency_benchmark.md ---
    retrieval_times = [r["retrieval_time_sec"] for r in results if r["retrieval_time_sec"] != ""]
    total_times = [r["total_time_sec"] for r in results if r["total_time_sec"] != ""]

    lines = [
        "# Latency Benchmark",
        "",
        "Min/max/avg computed across all successful queries in",
        "`query_set_ground_truth.csv` (20 queries), each run once through the",
        "live system -- this substitutes for the brief's original '5 queries x 3",
        "repeats' design, as agreed, since the two evaluations were combined into",
        "a single pass over a larger, more representative query set.",
        "",
        "| Metric | Min (s) | Max (s) | Avg (s) | N |",
        "|---|---|---|---|---|",
    ]
    if retrieval_times:
        lines.append(
            f"| Retrieval only | {min(retrieval_times):.4f} | {max(retrieval_times):.4f} | "
            f"{statistics.mean(retrieval_times):.4f} | {len(retrieval_times)} |"
        )
    if total_times:
        lines.append(
            f"| End-to-end (retrieval + generation) | {min(total_times):.4f} | {max(total_times):.4f} | "
            f"{statistics.mean(total_times):.4f} | {len(total_times)} |"
        )
    errored = [r["query_id"] for r in results if r["error"]]
    if errored:
        lines += ["", f"**Queries that errored (excluded from timing stats):** {', '.join(errored)}"]
    (PROJECT_ROOT / "latency_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote latency_benchmark.md")

    return results, llm_peak_rss_mb, llm_peak_ollama_rss_mb


# ---------------------------------------------------------------------------
# Step 4 — resource usage
# ---------------------------------------------------------------------------

def step4_resource_usage(embed_peak_rss_mb: float, llm_peak_rss_mb: float, llm_peak_ollama_rss_mb: float) -> None:
    print("\n=== STEP 4: Resource usage summary ===")
    ollama_rss_display = f"{llm_peak_ollama_rss_mb:.1f}" if llm_peak_ollama_rss_mb else "not detected"
    lines = [
        "# Resource Usage (non-rigorous, psutil-based)",
        "",
        "Sampled on a background thread every ~100ms via `psutil` while each",
        "phase ran. Not rigorous profiling -- coarse peak RSS / CPU snapshots",
        "only, per the brief's own caveat that this doesn't need to be precise.",
        "",
        "| Phase | Peak RSS -- Python process (MB) | Peak RSS -- Ollama server process (MB) |",
        "|---|---|---|",
        f"| Embedding generation (Step 2, ingestion) | {embed_peak_rss_mb:.1f} | n/a (embeddings run in-process via sentence-transformers) |",
        f"| LLM inference (Step 3, generation) | {llm_peak_rss_mb:.1f} | {ollama_rss_display} |",
    ]
    (PROJECT_ROOT / "resource_usage.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote resource_usage.md")


# ---------------------------------------------------------------------------
# Step 5 — summary
# ---------------------------------------------------------------------------

def step5_summary(results: list[dict]) -> None:
    print("\n=== STEP 5: Summary and sanity check ===")

    valid = [r for r in results if not r["error"]]
    errored = [r for r in results if r["error"]]

    total = len(results)
    hits = sum(int(r["hit"]) for r in valid)
    overall_accuracy = hits / total if total else 0.0

    by_type: dict[str, list[dict]] = {}
    for r in valid:
        by_type.setdefault(r["query_type"], []).append(r)

    retrieval_times = [r["retrieval_time_sec"] for r in valid]
    total_times = [r["total_time_sec"] for r in valid]

    lines = [
        "# Evaluation Summary",
        "",
        f"**Overall accuracy:** {hits}/{total} = {overall_accuracy:.1%}",
        "",
        "## Accuracy by query type",
        "",
        "| Query type | Hits | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for qtype, rows in sorted(by_type.items()):
        h = sum(int(r["hit"]) for r in rows)
        n = len(rows)
        lines.append(f"| {qtype} | {h} | {n} | {h/n:.1%} |")

    lines += ["", "## Latency (seconds)", "", "| Metric | Min | Max | Avg |", "|---|---|---|---|"]
    if retrieval_times:
        lines.append(
            f"| Retrieval only | {min(retrieval_times):.4f} | {max(retrieval_times):.4f} | {statistics.mean(retrieval_times):.4f} |"
        )
    if total_times:
        lines.append(
            f"| End-to-end | {min(total_times):.4f} | {max(total_times):.4f} | {statistics.mean(total_times):.4f} |"
        )

    lines += ["", "## Notes"]
    lines.append(
        "- `query_set_ground_truth.csv` lists `D8_Amazon_Rainforest_Briefing.wav` as the "
        "ground-truth doc, but the actual file on disk is `D8_Amazon_Rainforest_Briefing.mp3` "
        "(same audio, different container). Hit detection matches on filename stem "
        "(ignoring extension) so this does not affect accuracy."
    )
    if errored:
        lines.append(f"- {len(errored)} quer(ies) errored and were excluded from accuracy/latency stats: "
                      + ", ".join(f"{r['query_id']} ({r['error']})" for r in errored))
    else:
        lines.append("- No queries errored.")

    (PROJECT_ROOT / "eval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote eval_summary.md")

    print(f"\nOverall accuracy: {hits}/{total} = {overall_accuracy:.1%}")
    for qtype, rows in sorted(by_type.items()):
        h = sum(int(r["hit"]) for r in rows)
        print(f"  {qtype}: {h}/{len(rows)}")

    if errored:
        print("\n[FLAG] The following queries errored instead of returning a result:")
        for r in errored:
            print(f"  {r['query_id']}: {r['error']}")

    print("\nFirst 3 rows of precision_eval_results.csv:")
    for r in results[:3]:
        print(f"  {r['query_id']} | retrieved_docs={r['retrieved_docs']!r} | "
              f"hit={r['hit']} | generated_answer={r['generated_answer'][:80]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    store = VectorStore()
    step1_config_values(store)

    ingestion_rows, embed_peak_rss_mb = step2_ingest(store)

    results, llm_peak_rss_mb, llm_peak_ollama_rss_mb = step3_query_eval(store)

    step4_resource_usage(embed_peak_rss_mb, llm_peak_rss_mb, llm_peak_ollama_rss_mb)
    step5_summary(results)


if __name__ == "__main__":
    main()
