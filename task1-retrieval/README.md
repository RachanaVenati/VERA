# VERA — UniAgent 2026 Task 1 submission

A self-contained, offline retrieval system for TIRA. One execution processes one
supplied collection (German, English, or Hessian law). No API credentials or
outbound network are needed at runtime. Models are downloaded during the build.

## Submitted behavior

1. Build a language-specific BM25 index of the supplied documents.
2. Retrieve up to 1,000 documents per query using only the short `query` field.
3. Rerank up to 50 candidates using nonempty `original_query.description` (otherwise `query`) and a multilingual
   cross-encoder. Narrative and relevance judgments are not used.
4. Score long documents in overlapping chunks: 512 total tokens per pair,
   64 document tokens of overlap, maximum chunk logit as the document score.
5. Reuse scores for identical description/text pairs; preserve every original ID.
6. Append the remaining BM25 results in their original order. Final TREC scores
   are reciprocal ranks, avoiding comparisons between BM25 and cross-encoder
   scores. Original scores are retained in diagnostics and traces.

Semantic similarity groups are not used to delete records: the development
pilot grouped different Erasmus activities at cosine 0.9775. Those experimental
review and calibration commands remain in `../development/task1/`.

Optional semantic auditing runs **after BM25 candidate selection and before
description-based reranking**. Build with `--build-arg INCLUDE_SEMANTIC_MODEL=1`
and run with `--semantic-mode audit`. The image then bundles the pinned
multilingual E5-small model. The default is `--semantic-mode off`, with no
embedding model download or semantic computation. BM25 always uses `query`;
the cross-encoder uses nonempty `original_query.description`, falling back to
`query` when absent or blank; narrative is unused.

Audit mode adds `semantic-groups.jsonl.gz` and `semantic-review-pairs.csv`, and
logs embedding model calls in the event trace. Defaults are a provisional
grouping threshold of 0.95 and review floor of 0.90, adjustable through
`--semantic-threshold` and `--semantic-review-floor`. It preserves every ID and
reranks every original candidate; distinct texts never share relevance scores.
Audit cost is additional to the reranking budget and needs separate runtime
validation before submission. Label pairs and calibrate thresholds using
`../development/task1/SEMANTIC_REVIEW.md` before implementing any removal policy.

Group-aware ordering is an optional experiment after cross-encoder scoring:
`--ranking-strategy group-winners` ranks the best-scoring member of each group
first; `--ranking-strategy mmr --mmr-lambda 0.8` balances normalized relevance
and redundancy. Both require `--semantic-mode audit`, preserve all IDs, and
leave unscored documents in the BM25 tail. The default is `cross-encoder`.
Audit mode also writes `semantic-similarities.jsonl.gz` for all candidate pairs.
See `../development/task1/GROUP_RANKING.md` for the controlled comparison
workflow; it evaluates five policies from one shared inference run per dataset
without uploading software or automatically changing the submission policy.

A 240-second reranking budget per query reserves time for the five-minute case
limit. If it is reached, the completed prefix is reranked and remaining documents
retain BM25 order. Model batches check the budget between calls; this is not a
hard interruption of native inference. Budget-limited queries are reported in
`summary.json` and the event trace. Genuine inference errors fail the execution.

Missing or blank descriptions fall back to the short query for reranking. Use `--mode bm25` only
when intentionally running the lexical comparison system. Defaults can be changed
with `--candidate-k`, `--batch-size`, and `--case-budget-seconds`.

## Runtime files

| File | Purpose |
| --- | --- |
| `submission.py` | Integrated TIRA entry point and full-ranking merge |
| `baseline.py` | Language detection, BM25 indexing and retrieval |
| `query_fields.py` | Explicit short-query and description access |
| `candidates.py` | Select candidates and attach text/metadata |
| `cross_encoder.py` | Offline model inference and document chunking |
| `reranking.py` | Exact-text score reuse and candidate ordering |
| `event_logging.py` | Contract-shaped model, observation, error, and decision logs |
| `Dockerfile`, `requirements.txt` | Reproducible base image and pinned model/dependencies |
| `german-stopwords.txt`, `.NOTICE` | German stopwords and attribution |
| `UPSTREAM.md` | Source and model provenance |

## Build and run locally

From the repository root:

```sh
docker build --platform linux/amd64 --provenance=false \
  -t vera-task1-submission task1-retrieval
mkdir -p runs/submission-de
docker run --rm --network none --platform linux/amd64 --cpus 4 --memory 8g \
  -v "$PWD/datasets/retrieval-de-spot-check:/input:ro" \
  -v "$PWD/runs/submission-de:/output" \
  -e inputDataset=/input -e outputDir=/output vera-task1-submission
```

Use an empty output directory. TIRA supplies the same input/output environment
variables. Explicit `--dataset /input --output /output` arguments are also accepted.
The organizers provide four CPU cores and 40 GB RAM; local checks use 8 GB.

Outputs:

- `run.txt.gz`: the final six-column TREC ranking, up to 1,000 results per query.
- `run-trace.jsonl.log.gz`: model inputs (token pairs and text), chunk logits,
  observations, errors, and a final decision per query, with causal event IDs.
- `reranking-scores.jsonl.gz`: candidate scores/ranks and winning chunk indices.
- `summary.json`: configuration, counts, per-query timing, and budget limits.
- `index-ir-metadata.yml`, `retrieval-ir-metadata.yml`: resource metadata.

The trace follows the [official event logging contract](https://github.com/uniagent-webis-de/uniagent-code/blob/main/cikm26/event-logging-contract/README.md).
No candidate document-text dump or sample datasets are packaged in the image.

## TIRA dry run and submission

Use a current TIRA CLI under Python 3.10 or newer. From the repository root:

```sh
tira-cli code-submission \
  --path task1-retrieval \
  --task uniagent-2026 \
  --dataset retrieval-de-spot-check-20260816-training \
  --platform linux/amd64 \
  --build-args=--provenance=false \
  --command 'python3 /app/submission.py --dataset $inputDataset --output $outputDir' \
  --dry-run
```

Other spot-check IDs are `retrieval-en-spot-check-20260816-training` and
`retrieval-hessian-law-de-spot-check-20260902-training`.

For actual submission, register/authenticate with your TIRA participant token,
ensure Git is clean, then remove `--dry-run`. No upload is performed by a dry run.
Do not add credentials to this directory. No external service token is required
inside the submission container.

On Docker Desktop using the `local` log driver, TIRA 0.0.205's tracker probe
expects bytes but Docker SDK returns None. The development-only
`../development/task1/scripts/tira_cli_local.py` wrapper requests `json-file`
logging per helper container, without changing daemon settings or this image.
Use it with the Python environment containing TIRA in place of `tira-cli`.

Tests, verification reports, and local evaluation commands are in
[development/task1](../development/task1/README.md). Validation status is recorded
there; a successful pilot alone does not establish improvement over BM25.

## Validation status

All 52 Docker tests and all three official TIRA dry runs passed. Audits preserved
all 65 queries and 46,305 result rows, validated model inputs and causal traces,
and confirmed each case ends with a decision. See the
[validation report](../development/task1/reports/SUBMISSION_VALIDATION.md).
Staging/commit/sync approval was declined, so the working tree still needs a
reviewed commit before a real code submission. No upload has been performed.
