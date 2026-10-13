# Upstream provenance

Initial source: https://github.com/uniagent-webis-de/uniagent-code

Commit: `aaba38d87063016fa2b6b76fd43dc8b077ba7648`

Baseline directory: `cikm26/baselines/retrieval-baseline-pyterrier`.
Source, tests, and stopword attribution were copied unchanged. Requirements add
`trectools==0.0.50` for local retrieval evaluation inside Docker.
The Dockerfile pins the upstream image by digest and includes our spot-check
runner plus the official `cikm26/evaluation/evaluate_submission.py` evaluator.
The retrieval spot-check datasets were copied from the same commit.
