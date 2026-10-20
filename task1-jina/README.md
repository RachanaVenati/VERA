# Jina Nano: standalone offline Task 1 retrieval

This package retrieves documents with `jinaai/jina-embeddings-v5-text-nano`,
pinned to revision `8a7f00aac812071b69403df470f1038ec85f8925`. It is separate
from the existing BM25/cross-encoder submission. This is a dense-only candidate,
not a claim that Jina outperforms the existing system on these datasets.

## Retrieval flow

1. Load the bundled model, tokenizer and adapters locally; use four CPU threads.
2. Read each input `query`. Description and narrative are not required or used.
3. Embed queries with native `task="retrieval", prompt_name="query"`.
4. Split all corpus texts into 480-token windows with 64-token overlap. Preserve
   original document IDs and token spans. No document deduplication or cleanup.
5. Buffer at most 128 chunks; sort by length and encode in batches of 16 with
   native `prompt_name="document"`. This reduces padding; ranking logic is unchanged.
6. Normalize and write 768-dimensional vectors to disk with an explicit row map.
7. Read the saved vectors through a memory map and compute exact query/chunk
   cosine similarities. Give each document its maximum chunk similarity.
8. Return up to 1,000 unique documents per query, with document-ID tie breaking.

Maximum model input is 512 tokens in this tested configuration. The code checks
prefixed lengths before inference and refuses silent truncation. No BM25,
cross-encoder, semantic duplicate grouping or external API is used.

## Files

| File | Purpose |
|---|---|
| `submission.py` | TIRA command entry point |
| `jina_retrieval.py` | Chunking, model calls, batching, search orchestration, logs |
| `jina_vector_store.py` | Normalized disk vectors, manifest, memory-mapped exact search |
| `event_logging.py` | Gzip JSONL event traces |
| `Dockerfile` / `requirements.txt` | Isolated pinned model and CPU dependencies |

## Outputs and index reuse

`run.txt.gz` is the TREC ranking; `run-trace.jsonl.log.gz` records embedding
inputs, errors and per-query decisions; `summary.json` records timing/settings.
Shared corpus model calls use case ID `__index__`; final decisions use query IDs.

`index/embeddings.f32`, `index/chunks.jsonl`, and `index/manifest.json` contain
the persistent local index. The manifest records model revision, settings,
dimensions, row count and streaming SHA-256 of the supplied documents file.
Index files are built from the supplied corpus at runtime, not preloaded test
vectors. Roughly 307 MB is needed for 100,000 float32 vectors at 768 dimensions,
plus metadata. Query vectors and similarity batches stay in RAM.

For additional local queries against the identical corpus, use
`--reuse-index /path/to/index` and a fresh output directory. The model revision,
corpus checksum, prompt task and window settings must match. Reusing an index
never substitutes an index from another corpus. TIRA defaults to building a new
index from its mounted documents. Model/indexing time counts toward runtime.

## Build, validate and upload

```bash
docker build --platform linux/amd64 --provenance=false -t vera-task1-jina task1-jina
docker run --rm --network none --cpus 4 --memory 6g \
  -v /absolute/input:/input:ro -v /absolute/output:/output \
  vera-task1-jina --dataset /input --output /output
```

Dependencies and the pinned custom model code/adapters are downloaded only during
build. The Docker build then tests actual query and document embedding calls
under `RUN --network=none`. Runtime sets Hugging Face/Transformers offline mode.
CPU/BLAS thread counts are bounded. No API keys or hosted LLM are required.

`.github/workflows/validate-task1-jina.yml` builds the standalone image, tests
it, runs the complete Hessian sample with networking disabled, and evaluates it.
The manual `.github/workflows/submit-task1-jina.yml` builds and spot-checks this
package via the TIRA client, then uploads/registers it for team `vera` under
`uniagent-2026`, using the existing repository secret. Neither workflow changes
the current BM25/cross-encoder submission.

Registered TIRA command:

```text
python3 /app/submission.py --dataset $inputDataset --output $outputDir
```

The original Jina experiment has passed actual-model loading; complete retrieval
quality and runtime, and the standalone package's full Docker validation, must
be checked before calling this candidate submission-ready. The 480-token setup
is conservative, not an ablation-proven optimum. The upstream model license is
CC-BY-NC-4.0.
