# Configuration Values

| Parameter | Value |
|---|---|
| Chunk size | 400 words |
| Chunk overlap | 50 words |
| Chunking unit | word-based sliding window (see `ingestion/chunker.py:chunk_text`) |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Similarity metric | l2 (squared Euclidean) -- ChromaDB default, not overridden via hnsw:space |
| top-k (default) | 5 |
| top-k (summary queries) | 50 |
| Local LLM (via Ollama) | llama3.2:3b (family: llama, parameter_size: 3.2B, quantization: Q4_K_M) |
| Ollama base URL | http://localhost:11434 |
| Vector store | ChromaDB PersistentClient at `./data/chroma_db` |

## Prompt template

Built by `app/rag_pipeline.py:build_prompt()`. Each retrieved chunk is
numbered and tagged with its source citation, then wrapped in a
context block with an instruction to answer only from that context:

```python
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
```

Resulting prompt shape sent to the LLM:

```text
You are a document question-answering assistant. The context below contains
text extracted from uploaded documents (PDFs, images, audio, etc.). Answer
the question using ONLY the text in the context. The context is already
extracted and safe to use -- do not refuse or question its origin. If the
user asks what text is in a document or image, simply report the text from
the context verbatim.
If the context does not contain enough information, respond exactly with:
"I don't have enough information in the documents to answer that."

--- CONTEXT ---
[1] <source> | <page_or_timestamp>
<chunk text>
... (one block per retrieved chunk) ...
--- END CONTEXT ---

Question: <user query>

Answer:
```