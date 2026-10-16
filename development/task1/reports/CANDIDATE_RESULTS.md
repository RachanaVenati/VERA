> Historical development-stage report. See the [current submission](../../../task1-retrieval/README.md).

# Candidate-selection verification

All 29 Docker tests passed. All three spot checks completed with network disabled.

Every decompressed TREC run matched the original BM25 baseline exactly; nDCG@10 scores are unchanged.

| Dataset | Prepared candidates |
| --- | ---: |
| retrieval-de-spot-check | 1086 |
| retrieval-en-spot-check | 1250 |
| retrieval-hessian-law-de-spot-check | 750 |

Each query has up to 50 candidates; some German queries have fewer matches.

Verified unique query/document-ID pairs, attached text, descriptions, and source metadata. Legal identity fields and decision dates are retained.

Artifacts: `../../../runs/candidates-step2/<dataset>/candidates.jsonl.gz` (ignored by Git). Semantic duplicate removal and cross-encoder reranking are not active yet.
