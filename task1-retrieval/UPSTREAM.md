# Upstream provenance

Initial source: https://github.com/uniagent-webis-de/uniagent-code

Commit: `aaba38d87063016fa2b6b76fd43dc8b077ba7648`

Baseline directory: `cikm26/baselines/retrieval-baseline-pyterrier`.
Source, original tests, and stopword attribution were initially copied unchanged.
`baseline.py` now delegates short-query selection to `query_fields.py`, with
additional tests for separation from the description and narrative. Requirements add
`trectools==0.0.50` for local retrieval evaluation inside Docker.
`baseline.py` also prepares an intermediate candidate file through `candidates.py`
after writing the original BM25 run. Candidate-selection tests are our additions.
The Dockerfile pins the upstream image by digest and includes our spot-check
runner plus the official `cikm26/evaluation/evaluate_submission.py` evaluator.
The retrieval spot-check datasets were copied from the same commit.

The standalone semantic grouping modules, optional image, and additional tests
are VERA additions. The embedding model and revision are recorded in
`text_embeddings.py` and `Dockerfile.semantic`; the image includes the model card.

The cross-encoder adapter, reranking logic/CLI, optional reranking image, and
additional tests are VERA additions. Its pinned model revision is recorded in
`cross_encoder.py` and `Dockerfile.reranking`; the image includes the model card.
