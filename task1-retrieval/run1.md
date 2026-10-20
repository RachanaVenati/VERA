# Run 1: top-100 reranking, revised locally for missing descriptions

This revises the earlier top-50 run to rerank the top 100 candidates. It uses the shared
`query_fields.py` fallback already implemented locally: use a nonempty
`original_query.description`, otherwise use `query`. The previously uploaded
image is unchanged until this revision is rebuilt and submitted.

Run this configuration inside the submission image:

```bash
python3 /app/submission.py --dataset "$inputDataset" --output "$outputDir" \
  --candidate-k 100 --semantic-mode off --ranking-strategy cross-encoder
```

For a later GitHub Actions submission, select **candidate_k: 100** and
**semantic_mode: off** in `submit-task1.yml`. No upload is initiated by editing
this file.

Our run uses a **two-stage retrieval pipeline: BM25 candidate retrieval followed by multilingual cross-encoder reranking**.

1. **Process each collection separately.**  
   German university documents, English university documents, and Hessian law documents receive separate indexes and runs. A query searches only its supplied collection.

2. **Index complete documents with BM25.**  
   We index the supplied document text without additional cleaning, document splitting, or duplicate removal. PyTerrier applies language-specific tokenization, stopword removal, and stemming:
   
   | Language | Tokenization | Stemming | Stopwords |
   |---|---|---|---|
   | German | Terrier UTF | German | Supplied German list |
   | English | Terrier English | Porter | Terrier list |

   BM25 parameters use the library defaults; they were not tuned.

3. **Retrieve candidates using the short query.**  
   The `query` field retrieves **up to 1,000 documents** using BM25. The top **100** become reranking candidates.

4. **Rerank using the description, falling back to the query.**
   We use `original_query.description` when it is a nonempty string; if it is missing, null, or blank, we use `query` itself. We pair this effective reranking text with each candidate’s text and score it using:
   
   ```text
   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
   ```
   
   The narrative field is unused. We use the pretrained model directly, without task-specific fine-tuning. The cross-encoder jointly scores reranking-text–document pairs; it does **not** compute embedding cosine similarity.

5. **Handle long documents through overlapping chunks.**  
   Chunking happens **during reranking**, rather than before BM25 indexing:
   
   - Maximum paired sequence length: **512 tokens**, including the effective reranking text and special tokens.
   - The complete effective reranking text is repeated for each chunk.
   - Document chunks overlap by **64 tokens**.
   - Document relevance is the **maximum raw relevance logit** across its chunks.

   For effective reranking text \(q_d\) and document chunks \(c_i\):
   
   \[
   s(d,q_d)=\max_i f_{\mathrm{CE}}(q_d,c_i)
   \]

6. **Reuse scores only for identical inputs.**  
   Identical effective-reranking-text–document-text pairs share a cached score, but **all original document IDs remain**. Semantic duplicate removal and the proposed cosine threshold of 0.95 are **not part of this run**.

7. **Construct the final ranking.**  
   Reranked candidates appear first, followed by the remaining documents in their original BM25 order. Cross-encoder ties are resolved by BM25 rank, then document ID. Final TREC scores are \(1/\text{rank}\), preserving this order without combining incompatible BM25 and cross-encoder scores.

8. **Apply runtime controls and produce traces.**  
   CPU inference uses four PyTorch threads and batches of 16. Each query has a **240-second soft reranking budget**, checked between batches. If exhausted, only the fully scored prefix is reranked; unscored documents retain BM25 order afterward. Outputs include the ranking, model-call traces, candidate scores, and runtime summaries.

You can adapt this paragraph for the paper:

> We implement a two-stage retrieval system based on BM25 and multilingual cross-encoder reranking. Each collection is indexed independently using complete document texts and language-specific tokenization, stopword removal, and stemming. The short query retrieves up to 1,000 documents, of which the top 100 are reranked using the query description when available, otherwise the short query, and the pretrained mMARCO multilingual MiniLM cross-encoder. Long documents are divided into overlapping token windows, with a maximum paired input length of 512 tokens and a document overlap of 64 tokens. Document relevance is determined by the maximum cross-encoder logit across its windows. Identical reranking-text–document pairs reuse scores while preserving all document identifiers. The reranked prefix is followed by the remaining BM25 results. A per-query soft time budget limits reranking computation.

Historical validation note (predates this fallback revision; it is not a status report for the current upload): our recorded checks passed 52 tests and three dataset dry runs, but these do not establish an improvement over BM25. The GitHub run you shared passed the processing stage and failed during image upload, so it does not yet constitute a completed TIRA submission.
