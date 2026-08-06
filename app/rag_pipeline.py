"""
app/rag_pipeline.py

RAG prompt construction and answer generation via local Ollama.
"""

import sys
from pathlib import Path

import httpx
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from retrieval.vector_store import VectorStore

_NO_INFO = "I don't have enough information in the documents to answer that."

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Construct a grounded RAG prompt.

    Each chunk is presented as a numbered block with its source citation so
    the LLM can reference provenance. The instruction explicitly forbids the
    model from using outside knowledge.

    Args:
        query:            The user's question.
        retrieved_chunks: Output of VectorStore.search() —
                          list of {"text", "metadata", "score"}.

    Returns:
        A fully-formed prompt string ready to send to the LLM.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        header = f"[{i}] {meta.get('source', 'unknown')} | {meta.get('page_or_timestamp', '')}"
        context_blocks.append(f"{header}\n{chunk['text'].strip()}")

    context = "\n\n".join(context_blocks)

    return (
        "You are a document question-answering assistant. "
        "The context below contains text extracted from uploaded documents (PDFs, images, audio, etc.). "
        "Answer the question using ONLY the text in the context. "
        "The context is already extracted and safe to use — do not refuse or question its origin. "
        "If the user asks what text is in a document or image, simply report the text from the context verbatim.\n"
        f"If the context does not contain enough information, respond exactly with:\n"
        f'"{_NO_INFO}"\n\n'
        "--- CONTEXT ---\n"
        f"{context}\n"
        "--- END CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str = None,
) -> dict:
    """
    Run the full RAG pipeline: build prompt → call Ollama → format response.

    Skips the LLM entirely when no chunks are provided.

    Args:
        query:            The user's question.
        retrieved_chunks: Output of VectorStore.search().
        model:            Ollama model name to use. Defaults to OLLAMA_MODEL
                          from config.

    Returns:
        {
            "answer":  str,
            "sources": [{"source": str, "page_or_timestamp": str, "snippet": str}]
        }
    """
    if not retrieved_chunks:
        return {"answer": _NO_INFO, "sources": []}

    if not query or not query.strip():
        return {"answer": "Please enter a question.", "sources": []}

    prompt = build_prompt(query, retrieved_chunks)
    active_model = model or OLLAMA_MODEL

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=active_model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"].strip()
    except httpx.ConnectError:
        answer = (
            "Error: Cannot reach the Ollama server. "
            "Start it with `ollama serve` and ensure the model is pulled."
        )
    except ollama.ResponseError as e:
        answer = f"Error: Ollama returned an error — {e}"

    # Build deduplicated source list (preserve first-seen order)
    seen: set[tuple] = set()
    sources = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        key = (meta.get("source", ""), meta.get("page_or_timestamp", ""))
        if key not in seen:
            seen.add(key)
            snippet = chunk["text"].strip().replace("\n", " ")[:120]
            sources.append({
                "source":            key[0],
                "page_or_timestamp": key[1],
                "snippet":           snippet,
            })

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore()

    # Seed a few chunks (duplicates are silently skipped by store_chunks)
    store.store_chunks([
        {
            "text": "Photosynthesis is the process by which plants convert sunlight into glucose using chlorophyll.",
            "metadata": {"source": "biology.txt", "file_type": "txt", "page_or_timestamp": "page 1", "chunk_index": 0},
        },
        {
            "text": "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration.",
            "metadata": {"source": "biology.txt", "file_type": "txt", "page_or_timestamp": "page 2", "chunk_index": 1},
        },
        {
            "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889.",
            "metadata": {"source": "travel.txt", "file_type": "txt", "page_or_timestamp": "page 1", "chunk_index": 0},
        },
        {
            "text": "Python is a high-level programming language known for its readable syntax and large ecosystem.",
            "metadata": {"source": "cs.txt", "file_type": "txt", "page_or_timestamp": "page 1", "chunk_index": 0},
        },
    ])

    query = "How do plants produce energy?"
    print(f"Query : {query!r}\n")

    chunks = store.search(query, top_k=3)
    print(f"Retrieved {len(chunks)} chunk(s):")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c['score']}  {c['metadata']['source']} | {c['metadata']['page_or_timestamp']}")
    print()

    result = generate_answer(query, chunks)

    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} ({s['page_or_timestamp']})")
        print(f"    {s['snippet']!r}")
