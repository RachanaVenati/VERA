# Task 1: Retrieval

Starting point: the official UniAgent PyTerrier baseline, with BM25 as the default.
It reads the supplied dataset using TIRA's ir_datasets integration, indexes the
provided corpus with language-specific stemming and stopwords, and writes
`run.txt.gz` in TREC format, plus language and resource metadata.
See [upstream provenance](UPSTREAM.md).

## Modular implementation

See [PIPELINE.md](PIPELINE.md) for the staged plan. The first stage is implemented
in `query_fields.py`: BM25 explicitly reads `query`; `reranking_text()` exposes
`original_query.description` for the future cross-encoder and rejects missing
descriptions. Narrative is unused. Semantic grouping is available as a separate
command described below; description-based cross-encoder reranking is also
available as a separate command. Neither is automatically run by `baseline.py`.
All 65 sample queries were checked through the real loader: the explicit short
query matches the former `default_text()` input, and descriptions are available
for each query. The existing BM25 scores therefore remain a valid reference.

The second stage, `candidates.py`, prepares up to 50 BM25 hits per query in
`candidates.jsonl.gz`. Each record includes `qid`, `doc_id`, `bm25_rank` (starting
at 1), `bm25_score`, `query`, `description`, original `text`, and available source
`metadata`. This extra file is an intermediate artifact; `run.txt.gz` still
contains the original BM25 results, up to 1,000 per query. No semantic duplicate
removal or reranking is applied. With fewer matching documents, fewer candidates
are produced. Only selected document texts are retained in memory.

To change the candidate count, append `--candidate-k 100` to a `baseline.py`
invocation. Missing descriptions are exported as null; reranking will require
them when enabled. Raw metadata is available for local input directories with
`documents.jsonl` or `documents.jsonl.gz`; dataset-ID inputs export empty metadata
because the TIRA document loader exposes only ID and text.

## Semantic duplicate grouping (stage 3)

Build the optional image after the BM25 image:

```sh
docker build --platform linux/amd64 -t vera-task1 task1-retrieval
docker build --platform linux/amd64 -f task1-retrieval/Dockerfile.semantic \
  -t vera-task1-semantic task1-retrieval
```

The image installs CPU-only PyTorch and downloads a pinned revision of
`intfloat/multilingual-e5-small` at build time. Model loading and execution are
offline. Dependencies are recorded in `requirements-semantic.txt`.

For each dataset, run the separate grouping stage on its candidates:

```sh
docker run --rm --network none --platform linux/amd64 --cpus 4 --memory 8g \
  -v "$PWD/runs:/runs" --entrypoint python3 vera-task1-semantic \
  /app/group_duplicates.py \
  --candidates /runs/candidates-step2/retrieval-de-spot-check/candidates.jsonl.gz \
  --output /runs/semantic-groups/retrieval-de-spot-check --threshold 0.95
```

Use a fresh output directory. Outputs are `duplicate-groups.jsonl.gz` and
`summary.json`. Each group records the query ID, representative document ID,
every member ID, original BM25 rank, and cosine similarity to the representative.
The summary records model revision, threshold, group counts, and runtime.

`semantic_duplicates.py` contains grouping logic; `text_embeddings.py` contains
the model adapter and long-document pooling; `group_duplicates.py` is the CLI.
Grouping is per query. Every member must meet the threshold directly against
its representative, avoiding transitive chains. Exact text is embedded once
per execution, even when it appears in several query result sets.

Groups are suggestions for review: 0.95 is experimental and model-dependent.
Dates, numbers, exceptions, and negation can still differ at high similarity.
Original candidates and BM25 rankings are preserved. Known legal identities
must match for grouping; no newest-date rule is applied. Without source metadata,
legal identity checks cannot protect those pairs. Full text is covered through
480-token chunks and a token-count-weighted mean, but pooled similarity does
not prove all chunks are duplicates. The embedding model uses the same `query: `
prefix on both documents, as required for symmetric similarity by its
[model card](https://huggingface.co/intfloat/multilingual-e5-small).

## Docker build and tests

### Cross-encoder reranking (stage 4)

Build the BM25 and semantic images first, then:

```sh
docker build --platform linux/amd64 -f task1-retrieval/Dockerfile.reranking \
  -t vera-task1-reranking task1-retrieval
```

The image packages a pinned revision of
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, including German and
English). It uses the CPU libraries already installed in the semantic image.
The model card and weights are downloaded at build time; runtime is offline.

Run it on a prepared candidate set:

```sh
docker run --rm --network none --platform linux/amd64 --cpus 4 --memory 8g \
  -v "$PWD/runs:/runs" --entrypoint python3 vera-task1-reranking \
  /app/rerank.py \
  --candidates /runs/candidates-step2/retrieval-de-spot-check/candidates.jsonl.gz \
  --output /runs/reranked/retrieval-de-spot-check
```

Optionally add `--groups /runs/semantic-groups/retrieval-de-spot-check/duplicate-groups.jsonl.gz`
to annotate the results with groups from exactly the same candidate set. Every
candidate ID must be represented once in that file. Semantic group membership
does not transfer scores between different texts. Only identical text paired
with the identical description reuses a score; all original IDs remain ranked.

`cross_encoder.py` handles local inference and token chunks. `reranking.py`
handles score reuse, ordering, and TREC output. `rerank.py` is the CLI.
Input is the description, not the short query or narrative. Missing descriptions
fail explicitly. The full description repeats in each document chunk; overly
long descriptions fail rather than being silently truncated. Defaults are 512
total tokens per pair, 64 overlapping document tokens, and batches of 16 pairs.
These are configurable with `--max-length`, `--chunk-overlap`, and `--batch-size`.
The highest chunk relevance logit becomes the document score; it is not a
probability. Ties use original BM25 rank, then document ID. Maximum aggregation
can favor longer documents and needs quality evaluation.

Outputs are `reranked-candidates.jsonl.gz`, `run.txt.gz`, and `summary.json`.
Scored candidates retain original BM25 rank/score, text, description, metadata,
new rank/score, chunk count, best chunk index, and score-source document ID.
The TREC run contains only the supplied candidates (normally up to 50 per query).
The original up-to-1,000 BM25 run is preserved separately; merging its tail and
integrating the full submission entry point belong to the next stage. Use a
fresh output directory for each run.

### BM25 image

From the repository root, with Docker running:

```sh
docker build --platform linux/amd64 -t vera-task1 task1-retrieval
docker run --rm --platform linux/amd64 --entrypoint python3 vera-task1 -m unittest test_baseline
```

The upstream Dockerfile also runs pytest during the build. The upstream base
image supplies the Python/JVM dependencies. Build-time image downloads require
network access. Use linux/amd64 on Apple Silicon to match the submission platform.
No external API credentials are required for this baseline.

## Local spot-check runs

Run from the repository root:

```sh
mkdir -p runs
for dataset in retrieval-de-spot-check retrieval-en-spot-check retrieval-hessian-law-de-spot-check
do
  docker run --rm --platform linux/amd64 \
    -v "$PWD/datasets:/datasets:ro" -v "$PWD/runs:/runs" \
    --entrypoint python3 vera-task1 /app/baseline.py \
    --dataset "/datasets/$dataset" --wmodel BM25 --output "/runs/$dataset"
done
```

Qrels are for evaluation only. The system uses queries and corpus documents.
Evaluate each run against its matching dataset using the commands documented
in [datasets/README.md](../datasets/README.md).

To run all three datasets and evaluate nDCG@10 locally in one container:

```sh
mkdir -p runs/bm25
docker run --rm --network none --platform linux/amd64 --cpus 4 --memory 8g \
  -v "$PWD/datasets:/datasets:ro" -v "$PWD/runs:/runs" \
  --entrypoint python3 vera-task1 /app/run_spot_checks.py \
  --datasets /datasets --output /runs/bm25
```

Use an empty output directory for each experiment; existing outputs are not overwritten.
The runner fails on retrieval or evaluation errors and writes per-dataset
`evaluation.json` files and a combined `summary.json`. It uses the official
shared evaluator copied from the upstream commit recorded in `UPSTREAM.md`.
Spot-check queries and judgments are synthetic; their scores verify and compare
development runs, but do not establish performance on the hidden test set.

## Event logging

This statistical pipeline exposes no tools to a model and calls no LLM, so
`run-trace.jsonl.log.gz` is omitted under the
[official event logging contract](https://github.com/uniagent-webis-de/uniagent-code/blob/main/cikm26/event-logging-contract/README.md).
Resource usage is recorded with tirex_tracker. If model tools are added, their
calls must be logged according to that contract.

## TIRA submission

After Docker tests and all spot checks pass, commit the source so the Git working
tree is clean. Authenticate the TIRA CLI with your participant API key. From the
repository root, use the official baseline command with our submission path:

```sh
tira-cli code-submission \
  --path task1-retrieval \
  --task uniagent-2026 \
  --dataset retrieval-de-spot-check-20260816-training \
  --command '/app/baseline.py --dataset $inputDataset --wmodel BM25 --output $outputDir' \
  --dry-run
```

Remove `--dry-run` only for the actual submission. The English dataset ID is
`retrieval-en-spot-check-20260816-training`; the law dataset ID is
`retrieval-hessian-law-de-spot-check-20260902-training`.

## Validation status

The candidate-selection stage passed 29 Docker tests. All three original BM25 spot checks completed offline and
passed output-format validation. See [baseline results](BASELINE_RESULTS.md).
All three datasets were rerun with candidate preparation enabled. TREC outputs
match the baseline exactly; see [candidate verification](CANDIDATE_RESULTS.md).
The TIRA submission dry run remains pending because the local TIRA CLI runs under
incompatible Python 3.8; no submission has been uploaded.

Semantic stage verification: 37 tests passed in the optional image and three
offline five-candidate pilots completed. See [semantic results](SEMANTIC_RESULTS.md),
including an observed case of related pages grouped above 0.95. Full-set threshold
calibration remains pending. Cross-encoder reranking is now available separately.

Cross-encoder verification: all 45 Docker tests passed, and five-candidate offline
pilots completed for each dataset. See [reranking verification](RERANKING_RESULTS.md).
Full candidate-set benchmarking and final TIRA entry-point integration are pending;
no improvement over BM25 is claimed from these pilots.
