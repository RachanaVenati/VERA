# Task 1 submission readiness

## Completed

- The submission folder contains 14 runtime/packaging files. Tests, reports, and experimental tools were moved to `development/task1/`; obsolete Dockerfiles were removed.
- One self-contained Dockerfile builds from the pinned organizer image and packages the pinned multilingual cross-encoder. It no longer depends on locally built intermediate images.
- The integrated entry point runs BM25 from the short query, reranks up to 50 documents using the description, and appends the complete original BM25 tail. Every retrieved document ID is preserved.
- Exact description/text duplicates reuse scores; semantic similarity does not delete documents. Long text uses overlapping token chunks.
- Per-query reranking has a soft 240-second budget checked between batches; budget-limited queries retain the remaining BM25 tail. Genuine inference errors fail explicitly.
- Model calls (including failures), inputs, outputs, causal references, observations, and final decisions are logged in the official event-trace schema.
- All 52 Docker tests passed against the packaged runtime with network disabled.
- All three official TIRA code-submission dry runs passed, without uploading.

## Complete-dataset results

| Dataset | Queries | Final results | Runtime (minutes) | Budget-limited queries |
| --- | ---: | ---: | ---: | ---: |
| retrieval-de-spot-check | 25 | 10541 | 24.45 | 1 |
| retrieval-en-spot-check | 25 | 25000 | 22.03 | 0 |
| retrieval-hessian-law-de-spot-check | 15 | 10764 | 43.79 | 6 |

All 65 queries and 46,305 result rows passed independent auditing: IDs match the original BM25 runs, ranks/scores are valid, descriptions are used as model inputs, chunk pairs stay within 512 tokens, causal parents are valid, and every case ends in a final decision.

## Local tooling

The host TIRA installation used incompatible Python 3.8. A temporary Python 3.11 environment with TIRA 0.0.205 was used. Docker Desktop uses the `local` logging driver, which causes Docker SDK synchronous calls to return None in TIRA's tracker probe. The development-only wrapper requests json-file logging per helper container. It does not change the submission or daemon settings. Image builds use linux/amd64 with provenance disabled.

## Remaining steps and limits

The code is locally validated for TIRA, but Git staging/commit/sync approval was declined. Therefore changes remain local and the working tree is dirty. A real code submission requires committing the reviewed changes and a clean Git tree. No upload was performed. Actual upload also requires the participant's registered TIRA credentials; registry push/account authorization was not tested by these dry runs.

Additional relevance-score reevaluation was declined and was not retried through another route. These checks establish execution, format, ID preservation, trace compliance, and sample runtime; they do not establish ranking-quality improvement or hidden-test performance.

See `task1-retrieval/README.md` from the repository root for exact build, run, and TIRA commands. Raw final artifacts are in `runs/submission-validation/` (ignored by Git).
