# Semantic redundancy experiment

The default TIRA submission remains BM25 plus cross-encoder reranking.
`submission.py --semantic-mode audit` now connects detection between BM25
candidate selection and description-based reranking. Shared implementation
modules live in `task1-retrieval`; review/calibration CLIs remain in development.
Audit mode does not delete IDs or share scores between different texts. Do not
describe it as enabled in earlier runs.

For GitHub Actions, choose `audit` in the manual workflow's `semantic_mode`
selector. This bundles the embedding model at build time and passes the audit
flag to TIRA. Choose `off` to reproduce the existing algorithm. Audit mode adds
embedding time and trace/output size; full-model runtime validation is pending.

## Stages

1. Generate the BM25 top-50 candidate artifact using `baseline.py`.
2. Run `group_duplicates.py` on that artifact before the standalone `rerank.py`.
   It uses local multilingual E5-small document embeddings, normalized cosine,
   and a provisional grouping threshold of 0.95. Groups use the highest BM25
   candidate as representative and require direct representative similarity.
3. Review `review-pairs.csv`: all within-query pairs at cosine >= 0.90 are
   included, even pairs blocked by legal identity checks. Full texts, metadata,
   descriptions, ranks, and a stable query-level split are provided. The CSV
   lives under ignored `runs/`; do not commit corpus text to Git.
4. Label each pair `duplicate` (interchangeable information), `distinct`
   (different facts, conditions, scope, or decisions), or `uncertain`. Leave
   blank if not reviewed. A shared topic is insufficient. Prefer exhaustive
   labeling of this exported population; selective labeling biases metrics.
5. Calibrate on calibration queries and inspect held-out validation precision.
   The default precision target is 0.98 with at least five predicted positives
   in each split. This is a pilot criterion, not a confidence guarantee. With
   insufficient labels or failed validation, no threshold is recommended.
6. Only after review, test representative selection and its effect on nDCG@10,
   recall and runtime against the current pipeline. This step is not enabled.

Existing legal guards require equal ECLI, or matching court, case number and
decision date when ECLI is unavailable. These guards do not establish identical
text. Dates never select the newest record. Related legal decisions stay separate.

## Commands

Use the development environment with NumPy and Sentence Transformers installed,
and a locally downloaded, pinned E5-small model. No model downloads are performed
by these scripts. From the repo root, prepend runtime modules to PYTHONPATH:

```sh
PYTHONPATH=task1-retrieval:development/task1/scripts python \
  development/task1/scripts/group_duplicates.py \
  --candidates runs/candidates-step2/retrieval-de-spot-check/candidates.jsonl.gz \
  --model-path /path/to/local/multilingual-e5-small \
  --threshold 0.95 --review-floor 0.90 \
  --output runs/semantic-review/de

# Edit labels in review-pairs.csv, then:
PYTHONPATH=task1-retrieval:development/task1/scripts python \
  development/task1/scripts/calibrate_duplicates.py \
  --reviews runs/semantic-review/de/review-pairs.csv \
  --output runs/semantic-review/de/calibration.json
```

Keep collections separate. Metrics describe only labeled above-floor pairs,
not recall across all corpus duplicates. Do not use hidden test judgments for
threshold selection. Preserve the split and evaluate calibration choices on
independent data before changing submission behavior.

The existing `rerank.py --groups .../duplicate-groups.jsonl.gz` accepts groups
as annotations only; all candidates still receive their own scores except
identical description/text pairs. It produces candidate-only rankings, not the
integrated submission's full BM25-tail run.
