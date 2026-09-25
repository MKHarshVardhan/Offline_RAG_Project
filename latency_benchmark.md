# Latency Benchmark

Min/max/avg computed across all successful queries in
`query_set_ground_truth.csv` (20 queries), each run once through the
live system -- this substitutes for the brief's original '5 queries x 3
repeats' design, as agreed, since the two evaluations were combined into
a single pass over a larger, more representative query set.

| Metric | Min (s) | Max (s) | Avg (s) | N |
|---|---|---|---|---|
| Retrieval only | 0.0127 | 0.0299 | 0.0169 | 20 |
| End-to-end (retrieval + generation) | 4.2205 | 17.9088 | 6.7825 | 20 |