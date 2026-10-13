# VERA

UniAgent 2026 submissions developed in order: Task 1 retrieval, then Task 2 solving.

- `task1-retrieval/`: official PyTerrier BM25 starting baseline and tests.
- `task2-solving/`: reserved for the solving system.
- `datasets/`: official retrieval spot-check inputs and qrels.

Each task is submitted separately to TIRA from its own directory. Development and testing use Docker.
See [Task 1 instructions](task1-retrieval/README.md).

Current status: Task 1 Docker build passed 20 tests and all three offline spot checks passed output-format validation. See [BM25 results](task1-retrieval/BASELINE_RESULTS.md). TIRA submission dry run is pending.
