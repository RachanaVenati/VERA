# Task 1: Retrieval

Starting point: the official UniAgent PyTerrier baseline, with BM25 as the default.
It reads the supplied dataset using TIRA's ir_datasets integration, indexes the
provided corpus with language-specific stemming and stopwords, and writes
`run.txt.gz` in TREC format, plus language and resource metadata.
See [upstream provenance](UPSTREAM.md).

## Docker build and tests

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

Docker build passed all 20 tests. All three spot checks completed offline and
passed output-format validation. See [baseline results](BASELINE_RESULTS.md).
The TIRA submission dry run remains pending because the local TIRA CLI runs under
incompatible Python 3.8; no submission has been uploaded.
