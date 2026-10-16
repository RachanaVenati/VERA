"""Description-based candidate reranking with exact-text score reuse only."""

import gzip
import math
from pathlib import Path
from cross_encoder import RerankingBudgetExceeded


def rerank_candidates(records: list[dict], scorer, groups: list[dict] | None = None) -> list[dict]:
    """Preserve all IDs, score distinct description/text pairs, and assign ranks.

Semantic group mappings are annotations, not permission to transfer scores
between different texts. Identical text with the same description is scored
once. Each query's score ties are broken by original BM25 rank and document ID.
"""
    seen, descriptions = set(), {}
    for record in records:
        key = (record["qid"], record["doc_id"])
        if key in seen:
            raise ValueError(f"Repeated query/document ID: {key}")
        seen.add(key)
        description = record.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Query {record['qid']} is missing its description.")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Document {record['doc_id']} has empty text.")
        if record["qid"] in descriptions and descriptions[record["qid"]] != description:
            raise ValueError(f"Conflicting descriptions for query {record['qid']}.")
        descriptions[record["qid"]] = description
    representatives = {}
    if groups is not None:
        for group in groups:
            members = group["members"]
            if group["representative_id"] not in {m["doc_id"] for m in members}:
                raise ValueError("Semantic representative must be a group member.")
            for member in members:
                key = (group["qid"], member["doc_id"])
                if key in representatives:
                    raise ValueError(f"Candidate appears in more than one group: {key}")
                representatives[key] = group["representative_id"]
        if representatives.keys() != seen:
            raise ValueError("Semantic groups must cover exactly the input candidate IDs.")
    cache, output, limited_queries = {}, [], set()
    for record in sorted(records, key=lambda r: (r["qid"], r["bm25_rank"], r["doc_id"])):
        if record["qid"] in limited_queries:
            continue
        key = (record["description"], record["text"])
        if key not in cache:
            try:
                result = scorer.score(*key)
            except RerankingBudgetExceeded:
                limited_queries.add(record["qid"])
                continue
            if not math.isfinite(result["score"]):
                raise ValueError("Cross-encoder returned a non-finite document score.")
            cache[key] = (result, record["doc_id"])
        result, source_id = cache[key]
        output.append({**record, "rerank_score": result["score"],
                       "chunk_count": result["chunk_count"],
                       "best_chunk_index": result["best_chunk_index"],
                       "score_source_doc_id": source_id,
                       "semantic_representative_id": representatives.get(
                           (record["qid"], record["doc_id"]), record["doc_id"]),
                       })
    output.sort(key=lambda r: (r["qid"], -r["rerank_score"], r["bm25_rank"], r["doc_id"]))
    ranks = {}
    for record in output:
        ranks[record["qid"]] = ranks.get(record["qid"], 0) + 1
        record["rerank_rank"] = ranks[record["qid"]]
    return output


def write_reranked_run(records: list[dict], path: Path) -> None:
    """TREC output for the reranked candidate set, using original document IDs."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            qid, doc_id = str(record["qid"]), str(record["doc_id"])
            if not qid or not doc_id or any(c.isspace() for c in qid + doc_id):
                raise ValueError("TREC query and document IDs cannot be empty or contain whitespace.")
            stream.write(f"{qid} Q0 {doc_id} {record['rerank_rank']} "
                         f"{record['rerank_score']:.9g} vera-cross-encoder\n")
