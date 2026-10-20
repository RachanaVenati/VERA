# Pure vector retrieval experiment

Search the entire corpus using pinned multilingual E5-small embeddings. Query
inputs use only `query`, prefixed with `query: `. Document chunks use `passage: `.
Documents are split into 480-token windows with 64-token overlap. Exhaustive
cosine search runs over all chunks, with maximum chunk similarity determining
each document's score. Original IDs are preserved, and up to 1,000 unique
documents are returned per query. No BM25 candidates or cross-encoder scores
enter dense retrieval. Empty documents are counted and skipped.

This is an experiment, not a TIRA submission. It uses the model already bundled
by `INCLUDE_SEMANTIC_MODEL=1`, but intentionally does not reuse the symmetric
`query: ` document embedding policy from the duplicate audit.

`compare-task1-dense.yml` runs all three public collections on four CPU cores
and 6 GB RAM, with outbound network disabled during retrieval. Only document
and query files are mounted as input. Judgments are used afterward by the TIRA
evaluator to compare dense-only retrieval and fresh BM25-only retrieval.
Evaluation includes nDCG@10; summaries record the full corpus embedding and
search runtime and chunk counts. This experiment does not impose a per-query
budget on shared corpus embedding; runtime feasibility must be checked before
considering submission.

No approximate index is used: each encoded chunk batch is immediately compared
with every query vector. This avoids storing all chunk vectors and makes the
search exact for this embedding representation. Max-chunk aggregation can favor
long documents, which should be considered when interpreting legal results.

Model: `intfloat/multilingual-e5-small`, revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`.
[Model card](https://huggingface.co/intfloat/multilingual-e5-small).
