# Jina Nano full-corpus vector retrieval

An isolated experiment using `jinaai/jina-embeddings-v5-text-nano` at revision
`8a7f00aac812071b69403df470f1038ec85f8925`. Dependencies are separate from E5
and the TIRA submission. The pinned custom model code and task adapters are
bundled in the image. Its actual query encoder is smoke-tested during build.

Uses native `task="retrieval"`, with `prompt_name="query"` for queries and
`prompt_name="document"` for corpus chunks. Only `query` supplies query text.
The model adds its own prefixes; no E5 prefixes are added. Float32 CPU inference
uses four threads, 480-token chunks, 64-token overlap, and 512-token input cap
for this initial comparison. This does not test its full long-context capacity.

## Storage

The experiment writes a persistent, reusable index:

```text
runs/jina-vector-search/<collection>/
  index/
    embeddings.f32   # normalized float32 vectors, one row per chunk
    chunks.jsonl     # row -> original document ID and token span
    manifest.json    # model revision, dimension, row count, source hash, chunking
  run.txt.gz
  summary.json
  evaluation.json
```

Use `vector_store.search(index_directory, query_vectors)` to search saved vectors
without encoding documents again. Query vectors must come from the same pinned
model, retrieval task, dimensions, and query prompt. It uses memory-mapped exact
cosine search, max-pooling chunk scores per document. No vector database or
FAISS server is needed. Original IDs remain intact; top 1,000 are returned.

At 768 dimensions, 100,000 float32 vectors require approximately 307 MB, plus
chunk metadata and model weights. Index files stay in ignored `runs/`, outside
Git. Each collection has its own directory. The manifest identifies the source
corpus so incompatible or stale vectors can be detected; callers must check
model/corpus compatibility before reuse.

`compare-task1-jina.yml` runs one selected collection and retains rankings,
metrics, and its index in a downloadable GitHub artifact for seven days. Download
that artifact into the local path above to keep the vectors on the Mac; until
then they are stored on the runner and in the artifact, not on the Mac. Artifacts
are temporary storage, so download any index you want to keep.

The initial collection is Hessian law. Evaluate English and German university
collections separately before claiming general quality or runtime feasibility.
No TIRA submission is performed by this workflow. Full-model verification and
benchmarking happen on GitHub; local tests cover vector-file integrity and
max-chunk aggregation with synthetic vectors.

[Official model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
provides the retrieval task API and requirements. The model's license is
CC-BY-NC-4.0, as specified in its card.
