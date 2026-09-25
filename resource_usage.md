# Resource Usage (non-rigorous, psutil-based)

Sampled on a background thread every ~100ms via `psutil` while each
phase ran. Not rigorous profiling -- coarse peak RSS / CPU snapshots
only, per the brief's own caveat that this doesn't need to be precise.

| Phase | Peak RSS -- Python process (MB) | Peak RSS -- Ollama server process (MB) |
|---|---|---|
| Embedding generation (Step 2, ingestion) | 632.8 | n/a (embeddings run in-process via sentence-transformers) |
| LLM inference (Step 3, generation) | 635.8 | 147.2 |