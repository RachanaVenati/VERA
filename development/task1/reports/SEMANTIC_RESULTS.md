> Historical development-stage report. See the [current submission](../../../task1-retrieval/README.md).

# Semantic grouping verification

BM25 image: 35 Docker tests passed. Optional semantic image: 37 tests passed.

The actual pinned multilingual E5-small model ran with network disabled on a pilot of five BM25 candidates from the first query of each sample dataset. This is execution verification, not full-corpus validation or threshold calibration.

| Dataset | Candidates | Groups | Multi-member groups | Seconds |
| --- | ---: | ---: | ---: | ---: |
| retrieval-de-spot-check | 5 | 2 | 1 | 21.281 |
| retrieval-en-spot-check | 5 | 4 | 1 | 28.756 |
| retrieval-hessian-law-de-spot-check | 5 | 5 | 0 | 34.385 |

Verified every original query/document-ID pair occurs in exactly one group, and every member meets the 0.95 threshold against its representative.

## Important pilot finding

German Erasmus teaching-mobility and staff-training pages were grouped at about 0.9775 similarity, despite addressing different activities. Exact URL variants of each page also appeared. This demonstrates that cosine similarity alone can group related but non-interchangeable information. The stage therefore writes reviewable similarity groups and does not delete IDs or modify BM25 rankings.

Legal records with different known identities are not grouped; the law pilot retained all five records separately. This protection depends on available source metadata.

Long texts are processed in all 480-token chunks and pooled, rather than embedding only the beginning. The tokenizer may log a full-text length warning during splitting; only bounded chunks are passed to the model.

Model: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.

Outputs: `../../../runs/semantic-pilot/<dataset>/duplicate-groups.jsonl.gz` and `summary.json` (ignored by Git). Full candidate-set grouping and threshold calibration remain pending. Cross-encoder reranking is not yet implemented.
