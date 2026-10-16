> Historical development-stage report. See the [current submission](../../../task1-retrieval/README.md).

# Retrieval pipeline, implemented in stages

1. **Query fields (implemented):** `query_fields.py` separates the short query
   for BM25 from the description for the future cross-encoder. Narrative is
   unused. BM25 still returns up to 1,000 documents. Missing descriptions are
   allowed for BM25 but will fail explicitly when reranking requests them.
2. **Candidate selection (implemented):** `candidates.py` selects up to 50 BM25
   hits per query by default (`--candidate-k` changes this), attaches original
   text/IDs, query/description, and available raw source metadata, and writes
   `candidates.jsonl.gz`. Source metadata is read from a local documents JSONL
   file; if only a dataset ID is supplied, metadata is empty. Original BM25
   `run.txt.gz` remains unchanged. PyTerrier's zero-based rank is exported as a
   one-based `bm25_rank`. Candidate text is not chunked or deduplicated yet.
3. **Semantic duplicate groups (implemented standalone):** `text_embeddings.py`
   loads local multilingual E5-small and pools all 480-token chunks.
   `semantic_duplicates.py` groups per query at cosine threshold 0.95 by default.
   `group_duplicates.py` saves representative/member mappings and scores.
   Highest BM25 rank chooses representatives; dates never replace records.
   Known legal records need matching ECLI, or court/file/date. Without legal
   metadata, identity protection cannot be applied. All IDs remain intact.
   Inspect groups before changing ranking or reusing reranker scores.
4. **Cross-encoder reranking (implemented standalone):** `cross_encoder.py`
   scores description/document pairs using pinned multilingual mMARCO MiniLM.
   Every document token is covered in overlapping chunks; the maximum chunk
   logit is the document score. `reranking.py` orders candidates, preserves all
   IDs, and reuses scores only for identical description/text pairs. Semantic
   groups are annotations, since high similarity did not prove interchangeability
   in stage 3. `rerank.py` writes scored candidates and a candidate-only TREC run.
   Full-set CPU benchmarking and nDCG@10 comparison remain to be performed.
5. **Final output (planned integration):** write ranked original document IDs in
   the existing TREC format, recording the experiment configuration.

Each implemented stage has its own module and tests. BM25 still runs without
embedding dependencies. The optional semantic Docker image includes a pinned
embedding model downloaded at build time and loads it offline. The optional
reranking image adds a locally packaged cross-encoder. Final pipeline integration
and merging the original BM25 tail are not implemented yet.
