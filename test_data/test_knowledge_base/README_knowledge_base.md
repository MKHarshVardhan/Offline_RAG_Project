# Synthetic test knowledge base for the RAG paper evaluation

8 documents, one per format your paper claims to support, on 8 unrelated topics so retrieval has an unambiguous correct answer:

| File | Format | Topic |
|---|---|---|
| D1_GreenLeaf_Sustainability_Report.pdf | PDF | Corporate sustainability report |
| D2_Northwind_Fleet_Maintenance_Policy.pdf | PDF | Vehicle fleet maintenance policy |
| D3_Aurora_Health_Privacy_Guidelines.pdf | PDF | Healthcare data privacy policy |
| D4_Project_Phoenix_SRS.docx | DOCX | Software requirements spec |
| D5_Riverside_DataScience_Catalog.docx | DOCX | University course catalog |
| D6_Meridian_Bank_Fraud_Alert.png | Image (OCR) | Bank fraud notice |
| D7_CedarPoint_Safety_Rules.png | Image (OCR) | Amusement park safety rules |
| D8_Amazon_Rainforest_Briefing.wav | Audio (speech) | Rainforest conservation briefing (~43s) |

`query_set_ground_truth.csv` has 20 test queries, each already labeled with which document holds the correct answer. Half are near-exact keyword overlaps with the source text (easy case); half are paraphrased with different wording than the source (the case that's supposed to demonstrate semantic retrieval actually beating keyword matching — this is the evidence your paper currently asserts but never measures).

Q19 is deliberately the exact same query already described anecdotally in your Section V ("Is there any file that is related to Amazon?") — running it against this corpus reproduces that result in a way that's now part of a real evaluation set instead of a one-off anecdote.

## How to use this with the benchmarking brief

1. Upload all 8 files into your app to build the knowledge base (this replaces the step in `local_claude_code_brief.md` that assumed you already had documents).
2. Run each query from `query_set_ground_truth.csv` through the app.
3. For each query, record: which document(s) the system actually retrieved, and whether the ground_truth_doc appears in that list (hit = 1, miss = 0).
4. Fill those two columns into the CSV and compute:
   - Overall accuracy (hits / 20)
   - Accuracy split by query_type (keyword-overlap vs semantic) — this split is the number that actually supports your claim that semantic retrieval beats exact keyword matching. If your semantic-query accuracy is meaningfully close to your keyword-query accuracy, that's your evidence; if it drops off, that's an honest limitation worth reporting instead.
5. If you want a stronger comparison, also run the same 20 queries through a naive keyword search (e.g., simple substring/BM25 match against the raw chunk text) and report both numbers side by side — that gives you an actual quantitative ablation instead of the qualitative Table I currently in the paper.

Send the filled-in CSV (plus the other benchmark files from `local_claude_code_brief.md`) back here and I'll write it into Section V.
