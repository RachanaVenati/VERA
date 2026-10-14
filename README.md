# VERA

UniAgent 2026 submissions developed in order: Task 1 retrieval, then Task 2 solving.

- `task1-retrieval/`: official PyTerrier BM25 starting baseline and tests.
- `task2-solving/`: reserved for the solving system.
- `datasets/`: official retrieval spot-check inputs and qrels.

Each task is submitted separately to TIRA from its own directory. Development and testing use Docker.
See [Task 1 instructions](task1-retrieval/README.md).

Current status: Task 1 has modular query handling, BM25 candidate preparation, semantic grouping, and description-based cross-encoder reranking. The optional reranking image passed 45 tests and three offline real-model pilots. See [reranking verification](task1-retrieval/RERANKING_RESULTS.md). Full-set benchmarking and final submission integration remain pending. All three original offline BM25 spot checks passed output-format validation. See [BM25 results](task1-retrieval/BASELINE_RESULTS.md) and the [modular pipeline plan](task1-retrieval/PIPELINE.md). TIRA submission dry run is pending.
