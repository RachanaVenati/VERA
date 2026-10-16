> Historical development-stage report. See the [current submission](../../../task1-retrieval/README.md).

# Cross-encoder reranking verification

The optional reranking Docker image passed all 45 tests. The real pinned multilingual mMARCO MiniLM cross-encoder completed three offline pilots with four CPU cores and an 8 GiB container limit.

Each pilot used the same five candidates and semantic groups as stage 3, covering one query per dataset. All 15 original IDs were preserved. Ranking order, finite scores, exact-text score reuse, and six-column TREC serialization were verified.

| Dataset | Candidates | Distinct description/text pairs scored | Score reuses | Seconds |
| --- | ---: | ---: | ---: | ---: |
| retrieval-de-spot-check | 5 | 3 | 2 | 10.664 |
| retrieval-en-spot-check | 5 | 4 | 1 | 17.844 |
| retrieval-hessian-law-de-spot-check | 5 | 5 | 0 | 31.305 |

## What changed

BM25 still retrieves using the short query. The new standalone cross-encoder stage scores candidates using the description. It covers full document text with overlapping token chunks and uses the maximum chunk relevance logit as the document score.

Erasmus teaching and staff-training pages previously grouped at cosine 0.9775 were scored separately: about 1.3777 versus -1.3107 in the German pilot. Identical URL variants reused their scores. This demonstrates separation of different content, not correctness of the resulting relevance order. Late chunks also supplied the highest scores for several documents.

## Limits and next stage

This is a pilot execution check, not a full-set nDCG@10 comparison. No retrieval improvement is claimed. Full-set runtime/quality benchmarking remains pending. Maximum chunk scoring can favor long documents.

Outputs contain only supplied candidates (normally at most 50 per query), with every original candidate ID retained. The up-to-1,000 BM25 tail is not merged. Production submission entry-point integration belongs to stage 5. Semantic similarity groups are annotations and never transfer a score between different texts.

Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, revision `1427fd652930e4ba29e8149678df786c240d8825`. Defaults: 512 tokens per pair, 64 overlapping document tokens, batch size 16, CPU inference, no runtime model downloads.

Outputs: `../../../runs/reranking-pilot/<dataset>/reranked-candidates.jsonl.gz`, `run.txt.gz`, and `summary.json` (ignored by Git).
