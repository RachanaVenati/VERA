# Compare group-aware ordering with plain cross-encoder ranking

The pipeline keeps `query` for BM25 and `original_query.description` for
cross-encoder relevance. Semantic embeddings compare document contents, not
query relevance. Ranking policies are applied after individual candidate scores
are computed. They change ordering, preserve all original IDs, and append the
untouched BM25 tail. They do not remove documents or transfer relevance scores.

## Policies

- `cross-encoder` (default): descending raw relevance logit, then BM25 rank and
  document ID for ties.
- `group-winners`: choose the highest-relevance scored member of each semantic
  group, sort winners by relevance, then append remaining scored members in
  relevance order. The grouping representative does not have to be the winner.
  If time limits leave group members unscored, only scored members participate;
  all unscored documents stay in the BM25 tail.
- `mmr`: greedily maximize `lambda * normalized_relevance - (1-lambda) *
  maximum_similarity_to_selected`. Relevance is min-max normalized over the
  scored candidates for the query (all equal becomes 1); similarity is cosine
  clipped below at zero. Legally incompatible pairs have zero redundancy
  penalty. Score ties retain relevance/BM25/ID order. Default lambda is 0.8.
  All within-query similarities are stored, including those below the audit
  review floor. Missing similarity pairs fail rather than imply zero.

Runtime options are `--ranking-strategy cross-encoder|group-winners|mmr` and
`--mmr-lambda 0.8`. Experimental strategies require `--semantic-mode audit` and
an image built with `--build-arg INCLUDE_SEMANTIC_MODEL=1`.

## Controlled comparison

Run **Actions → Compare Task 1 ranking policies → Run workflow → main**.
This workflow needs no TIRA token, uploads no software to TIRA, and launches
independent jobs for the three public sample collections. Each job:

1. Builds the image and runs ranking/field-separation tests.
2. Runs BM25, document embeddings and cross-encoder scoring once with no qrels
   mounted into the inference container.
3. Uses those frozen scores and candidate IDs to generate the default ordering,
   group-winner ordering, and MMR at lambda 0.7, 0.8 and 0.9.
4. Verifies that reconstructing the default ordering reproduces the original
   run exactly and every policy preserves the full ID set.
5. Computes nDCG@10 through the existing TIRA evaluator and independently
   computes Recall@1000 over positive judgments. Recall should be identical
   because only order changes.
6. Reports fixed query-level calibration and validation splits (SHA256 of query
   ID modulo five assigns one bucket to validation, as in semantic review).
   Chooses a policy using calibration nDCG@10 only, with the default winning
   ties, and reports that policy's held-out performance. Full-sample results
   are descriptive, not a separate held-out test. Samples/splits are small;
   do not claim a robust improvement from these alone.

Download `ranking-comparison-<collection>` from workflow artifacts. It contains
`comparison.json`, a Markdown results table, each policy's complete run,
the source configuration and timing, scores, group mappings and similarities.
It excludes full document review text and model traces to keep artifacts small.
No policy is automatically deployed. The shared inference runtime is recorded
once; per-policy reordering time excludes inference and evaluation.

For a completed audit run with the new all-pairs artifact, comparison alone is:

```sh
PYTHONPATH=task1-retrieval:development/task1/scripts python \
  development/task1/scripts/compare_group_rankings.py \
  --source runs/audit-de --truths datasets/retrieval-de-spot-check \
  --output runs/group-ranking-comparison/de
```

Use the submission container's dependencies for official evaluation. Old audit
runs without `semantic-similarities.jsonl.gz` must be regenerated. An earlier
time-limited run can compare policies over its scored prefix, but no policy
can promote unscored candidates into that prefix.
