# Ingestion Benchmark

Single run per file (not averaged over 3 runs as in the original brief --
the evaluation was combined into one ingestion pass per the user's
instructions, to avoid ingesting the same documents multiple times;
ChromaDB's content-hash dedup would make repeat runs no-ops without a
reset between each one anyway).

| Format | File | Size (KB) | Chunk count | Extraction time (s) | Embedding+store time (s) |
|---|---|---|---|---|---|
| pdf | D1_GreenLeaf_Sustainability_Report.pdf | 2.4 | 1 | 2.6069 | 0.1865 |
| pdf | D2_Northwind_Fleet_Maintenance_Policy.pdf | 2.5 | 1 | 1.5556 | 0.0725 |
| pdf | D3_Aurora_Health_Privacy_Guidelines.pdf | 2.3 | 1 | 1.2482 | 0.0713 |
| docx | D4_Project_Phoenix_SRS.docx | 36.2 | 1 | 0.0477 | 0.0598 |
| docx | D5_Riverside_DataScience_Catalog.docx | 36.2 | 1 | 0.0333 | 0.0742 |
| png | D6_Meridian_Bank_Fraud_Alert.png | 84.1 | 1 | 1.2795 | 0.0613 |
| png | D7_CedarPoint_Safety_Rules.png | 76.9 | 1 | 1.2418 | 0.1016 |
| mp3 | D8_Amazon_Rainforest_Briefing.mp3 | 716.8 | 7 | 5.9833 | 0.0632 |

**Total documents ingested:** 8  
**Total chunks in vector store:** 14  
**Peak RSS during embedding (this process, coarse sampling):** 632.8 MB