# Retrieval pipeline, implemented in stages

1. **Query fields (implemented):** `query_fields.py` separates the short query
   for BM25 from the description for the future cross-encoder. Narrative is
   unused. BM25 still returns up to 1,000 documents. Missing descriptions are
   allowed for BM25 but will fail explicitly when reranking requests them.
2. **Candidate selection (next):** select a configurable number of BM25 hits
   per query, and attach their original text and document IDs.
3. **Semantic duplicate groups (planned):** embed candidates and compare cosine
   similarity using a configurable threshold, initially 0.95. Keep a mapping of
   every original ID to its representative; do not select legal records solely
   by date. Inspect groups before deciding whether to remove IDs from output.
4. **Cross-encoder reranking (planned):** score description–text pairs with a
   multilingual model. Handle long text explicitly through chunking and document
   score aggregation. Benchmark CPU runtime and nDCG@10 against BM25.
5. **Final output (planned integration):** write ranked original document IDs in
   the existing TREC format, recording the experiment configuration.

Each stage will have its own module and tests. No embedding or cross-encoder
model is loaded in the current program. Model choice, candidate count, chunk
handling, and duplicate-output policy will be made explicit when those stages
are implemented. Models must be available locally during offline execution.
