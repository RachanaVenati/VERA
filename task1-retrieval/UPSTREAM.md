# Provenance

The BM25 starting point is the organizers' PyTerrier baseline:
https://github.com/uniagent-webis-de/uniagent-code/tree/main/cikm26/baselines/retrieval-baseline-pyterrier

Original source commit: `aaba38d87063016fa2b6b76fd43dc8b077ba7648`.

VERA adds explicit query/description separation, candidate preparation,
description-based cross-encoder reranking, exact-text score reuse, complete
BM25-tail merging, runtime budgeting, and UniAgent event traces. Indexing streams
documents rather than materializing the full corpus.

Cross-encoder: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, revision
`1427fd652930e4ba29e8149678df786c240d8825` (Apache-2.0). The model card is included
in the image beside the weights. German stopword attribution is preserved in
`german-stopwords.NOTICE`.

Development tests, historical reports, and experiments live in
`../development/task1/` and are outside this Docker build context.
