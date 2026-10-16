> Historical development-stage report. See the [current submission](../../../task1-retrieval/README.md).

# BM25 baseline results

Docker build: 20 tests passed. All three datasets completed with network disabled.

Local limits: four CPU cores and 8 GiB RAM, linux/amd64 on Apple Silicon.

| Dataset | nDCG@10 |
| --- | ---: |
| retrieval-de-spot-check | 0.697856 |
| retrieval-en-spot-check | 0.762747 |
| retrieval-hessian-law-de-spot-check | 0.399989 |

All three runs passed the official TIRA evaluator's output-format validation.

Raw runs, resource metadata, and evaluation JSON are in `../../../runs/bm25-offline/` (ignored by Git).

These spot-check queries and judgments are synthetic. Do not interpret these scores as hidden-test performance.

TIRA submission dry run is pending: the host CLI is installed under Python 3.8 and crashes on import; Python 3.11 has no TIRA CLI installed. No submission was uploaded.
