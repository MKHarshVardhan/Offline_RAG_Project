# Evaluation Summary

**Overall accuracy:** 20/20 = 100.0%

## Accuracy by query type

| Query type | Hits | Total | Accuracy |
|---|---|---|---|
| keyword-overlap | 12 | 12 | 100.0% |
| semantic | 8 | 8 | 100.0% |

## Latency (seconds)

| Metric | Min | Max | Avg |
|---|---|---|---|
| Retrieval only | 0.0127 | 0.0299 | 0.0169 |
| End-to-end | 4.2205 | 17.9088 | 6.7825 |

## Notes
- `query_set_ground_truth.csv` lists `D8_Amazon_Rainforest_Briefing.wav` as the ground-truth doc, but the actual file on disk is `D8_Amazon_Rainforest_Briefing.mp3` (same audio, different container). Hit detection matches on filename stem (ignoring extension) so this does not affect accuracy.
- No queries errored.