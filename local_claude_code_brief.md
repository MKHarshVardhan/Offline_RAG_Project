# Brief for local Claude Code — RAG paper evaluation data

Paste this whole file into your Claude Code session in VS Code (the one with the RAG project open). Goal: produce data + artifacts needed to strengthen the "System Implementation" and "Experimental Results" sections of an IEEE paper. Do NOT change the paper itself — only inspect/run the code and produce the outputs listed below.

## 1. Extract exact configuration values

Search the codebase for and report the actual values currently used for:
- Chunk size and chunk overlap (characters or tokens) used when splitting documents
- Embedding model name + embedding dimension (e.g., all-MiniLM-L6-v2, 384-dim)
- Similarity metric used for retrieval (cosine / L2 / dot product) — check the ChromaDB collection config
- top-k value (how many chunks are retrieved per query)
- The exact local LLM name + parameter size + quantization used via Ollama (e.g., llama3.2:3b-instruct-q4_0)
- Any prompt template used to combine retrieved chunks + query before sending to the LLM

Output as a simple table (Parameter | Value) in a file called `config_values.md`.

## 2. Benchmark ingestion + embedding performance

For each supported format (PDF, DOCX, image/OCR, audio) that you have at least one sample file for:
- Time how long ingestion + text extraction takes for one representative file
- Time how long embedding generation takes for that file's resulting chunks
- Record file size and resulting chunk count

Run each timing 3 times and report the average. Output as a table in `ingestion_benchmark.md` (columns: Format | File size | Chunk count | Extraction time (s) | Embedding time (s), avg of 3 runs).

## 3. Retrieval + end-to-end latency benchmark

Pick 5 representative queries against the existing indexed knowledge base. For each:
- Time the retrieval step alone (query embedding + ChromaDB search)
- Time the full end-to-end response (retrieval + LLM generation)
- Run each 3 times, report min/max/avg

Output as `latency_benchmark.md`.

## 4. Precision@k evaluation set (the most important part)

Build a small labeled test set: 15–20 natural-language questions where you (the developer) know which specific uploaded document(s) contain the correct answer. Mix easy (near-exact keyword match) and hard (paraphrased, no shared keywords) queries.

For each query, run it through the system and record:
- The query text
- The document(s) that SHOULD be retrieved (ground truth, decided by you before running)
- The document(s)/chunks actually retrieved by the system (top-k)
- Whether the correct document appeared in the top-k results (yes/no)
- The generated answer's correctness (your own judgment: correct / partially correct / incorrect)

Compute: Precision@k, and simple accuracy (% of queries where the correct source was retrieved).

Output as a CSV `precision_eval.csv` with columns: query, ground_truth_doc, retrieved_docs, hit(0/1), answer_quality, plus a one-line summary of the aggregate Precision@k and accuracy at the top.

## 5. Screenshots

Take and save as PNG:
- The document upload / knowledge base creation screen with a few files loaded
- A query submitted with the generated answer AND the source document citation visible in the same screenshot
- (optional) A screenshot showing a query that fails or returns a partial answer — useful for the Limitations section

## 6. Resource usage (optional but valuable)

If easy to grab: peak RAM usage and CPU utilization during embedding generation and during LLM inference (Task Manager / `htop` / Python's `psutil` is fine — doesn't need to be rigorous profiling).

## Deliverables to send back

- `config_values.md`
- `ingestion_benchmark.md`
- `latency_benchmark.md`
- `precision_eval.csv`
- Screenshots (PNG)
- (optional) resource usage notes

Zip these together if easiest, or just attach them individually.
