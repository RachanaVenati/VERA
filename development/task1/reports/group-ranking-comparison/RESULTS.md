# Group-ranking comparison results

Recovered evaluation: https://github.com/RachanaVenati/VERA/actions/runs/39332478846

Inference source: https://github.com/RachanaVenati/VERA/actions/runs/39327426915

Public sample datasets evaluated on GitHub using the TIRA metric engine. These are not remote TIRA test-set scores. Each policy uses the same saved cross-encoder scores; all document IDs are retained. Small split evaluation uses TIRA parsers and its metric method without full-dataset minimum-size assumptions. Full-collection evaluation retains normal checks.

| Collection | Cross-encoder | Group winners | MMR 0.7 | MMR 0.8 | MMR 0.9 |
|---|---:|---:|---:|---:|---:|
| retrieval-de-spot-check | 0.874078 | 0.861950 | 0.874557 | 0.874557 | 0.874078 |
| retrieval-en-spot-check | 0.835165 | 0.804723 | 0.835165 | 0.835165 | 0.835165 |
| retrieval-hessian-law-de-spot-check | 0.565295 | 0.565295 | 0.565295 | 0.565295 | 0.565295 |

The calibration-selected German MMR variant has no held-out improvement over plain cross-encoder ranking. English and Hessian-law calibration select the plain cross-encoder baseline. Group-winner ordering decreases full-sample nDCG@10 on German and English. These small samples do not support deploying a group-aware ordering change. The default submission policy remains cross-encoder.

Validation split sizes: German 5 queries, English 2, Hessian law 2. Recall@1000 remains identical across all policies: German 1.0, English 1.0, Hessian law 0.933333. No additional model inference was needed for recovery.
