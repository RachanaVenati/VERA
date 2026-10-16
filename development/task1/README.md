# Task 1 development and validation

`../../task1-retrieval/` is the submission build context. This directory keeps
unit tests, evaluation utilities, experiments, and historical reports outside
that context. Superseded Dockerfiles were removed; their versions remain in Git history.
The current submission uses one self-contained Dockerfile.

## Run all tests against the packaged runtime

From the repository root, after building `vera-task1-submission`:

```sh
docker run --rm --network none --platform linux/amd64 --cpus 4 --memory 8g \
  -e VERA_RUNTIME_PATH=/app -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" --entrypoint python3 vera-task1-submission \
  -m pytest -p no:cacheprovider /workspace/development/task1/tests
```

## Full spot checks and official evaluation

```sh
mkdir -p runs/submission-validation
docker run --rm --network none --platform linux/amd64 --cpus 4 --memory 8g \
  -v "$PWD/datasets:/datasets:ro" -v "$PWD/runs:/runs" \
  -v "$PWD/development/task1/scripts:/development:ro" \
  --entrypoint python3 vera-task1-submission /development/run_spot_checks.py \
  --datasets /datasets --output /runs/submission-validation
```

Use fresh output directories. The runner uses the original shared evaluator,
which remains in `scripts/evaluate_submission.py`. `--dataset-name` can select
one or more collections. Sample qrels are evaluation-only and never read by the
submission. Historical stage results are in `reports/`; they are not a claim
about the final integrated system or hidden-test performance.

## Local TIRA CLI compatibility

A temporary Python 3.11 CLI environment is available at
`/private/tmp/vera-tira-cli/` for this session. The previous host CLI uses Python
3.8 and cannot import its installed TIRA version. For Docker Desktop's `local`
logging driver, run the CLI via `scripts/tira_cli_local.py` so synchronous helper
containers use json-file logging. This is a local tooling workaround, not a
submission-runtime patch.

To recreate the tested CLI helper without changing the host's Python setup:

```sh
PYENV_VERSION=3.11.9 python -m venv /private/tmp/vera-tira-cli
/private/tmp/vera-tira-cli/bin/pip install -r development/task1/requirements-cli.txt
/private/tmp/vera-tira-cli/bin/python development/task1/scripts/tira_cli_local.py \
  code-submission --path task1-retrieval --task uniagent-2026 \
  --dataset retrieval-de-spot-check-20260816-training --platform linux/amd64 \
  --build-args=--provenance=false \
  --command 'python3 /app/submission.py --dataset $inputDataset --output $outputDir' \
  --dry-run
```

After execution, `scripts/audit_submission.py` checks TREC ranks, ID coverage,
model-input descriptions, chunk limits, trace schema/causal links, and final
decisions without reading qrels:

```sh
python3 development/task1/scripts/audit_submission.py \
  --output runs/submission-validation/retrieval-de-spot-check \
  --dataset datasets/retrieval-de-spot-check
```

Final status: [submission validation](reports/SUBMISSION_VALIDATION.md) and
[machine-readable audits](reports/SUBMISSION_VALIDATION.json).
