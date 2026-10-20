# Top-50 versus top-100 candidate experiment

Run **Actions → Compare Task 1 candidate pools → Run workflow**. The default
collection is Hessian law; German and English university collections are also
available. This workflow does not upload or execute software on TIRA.

Two independent fresh runs use the same code, pretrained cross-encoder,
language analyzer, short `query` for BM25, `original_query.description` for
reranking, batch size 16, four CPU cores, six GB container memory, and a
240-second soft reranking budget per query. Only `--candidate-k` changes from
50 to 100. Semantic audit/grouping and diversity ordering are disabled in both
runs, so only the cross-encoder candidate-pool size is being tested. Models
are local and inference has no network or qrels access.

The ranking still contains up to 1,000 original document IDs. The completed
scored prefix is reranked, with the unscored tail retaining BM25 order. A
larger requested pool does not guarantee every candidate is scored if a query
hits its time budget.

The report distinguishes:

- Candidate recall: fraction of judged positive documents in the requested
  BM25 top-k pool, regardless of whether scoring finished.
- Scored recall: fraction of judged positives actually scored by the
  cross-encoder before its budget expired.
- Final nDCG@10: computed with the existing full-collection TIRA evaluator.
- Recall@1000, scored document counts, per-target original/final ranks,
  end-to-end time, per-query scoring time, and budget-limited query IDs.

The summary job checks model/configuration compatibility and identical original
BM25 ranks for the judged targets. It publishes `candidate-pool-comparison`
with a Markdown table and JSON reports. Each job also retains its complete
ranking and score diagnostics. Hardware timing may vary across independent
runners. Public sample scores are exploratory; the default submission remains
top 50 until the measured tradeoff is reviewed.

Based on existing Hessian BM25 ranks, top 100 adds the positive target for
query 15 at rank 96. Targets at ranks 148, 492, and outside the top 1,000 remain
outside the requested pool. Their retrieval improvement needs a separate
candidate-generation experiment.
