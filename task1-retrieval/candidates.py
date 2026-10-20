"""Select BM25 candidates and attach text for later deduplication/reranking.

This stage does not change scores, deduplicate content, or use relevance truths.
Only text belonging to selected document IDs is retained in memory.
"""

import gzip
import json
from pathlib import Path

import pandas as pd

from query_fields import retrieval_text, reranking_text, reranking_field


def select_candidates(run: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Take the first top_k distinct document IDs per query in BM25 rank order."""
    if top_k < 1:
        raise ValueError("Candidate count must be at least 1.")
    required = {"qid", "docno", "rank", "score"}
    if not required.issubset(run.columns):
        raise ValueError("BM25 results require qid, docno, rank, and score columns.")
    ordered = run.copy()
    ordered["qid"] = ordered["qid"].astype(str)
    ordered["docno"] = ordered["docno"].astype(str)
    return (
        ordered.sort_values(["qid", "rank"], kind="stable")
        .drop_duplicates(["qid", "docno"])
        .groupby("qid", sort=False).head(top_k).reset_index(drop=True)
    )


def _read_metadata(directory: Path | None, wanted: set[str]) -> dict:
    """Preserve available source metadata, including legal identity and date."""
    if directory is None:
        return {}
    path = next((directory / name for name in ("documents.jsonl.gz", "documents.jsonl")
                 if (directory / name).is_file()), None)
    if path is None:
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    metadata = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            doc_id = str(document["doc_id"])
            if doc_id in wanted:
                metadata[doc_id] = {key: value for key, value in document.items()
                                    if key not in {"doc_id", "text", "content"}}
    return metadata


def prepare_candidates(dataset, run: pd.DataFrame, top_k: int = 50,
                       source_directory: Path | None = None) -> list[dict]:
    """Attach original IDs/text and separate query/description to selected hits.

The description key contains the effective reranking text: a nonempty original
description, otherwise the query. Missing candidate documents fail explicitly.
Metadata is available when a local source documents JSONL file is supplied;
otherwise it is empty, since the TIRA document loader exposes only ID and text.
"""
    selected = select_candidates(run, top_k)
    wanted = set(selected.docno)
    if not wanted:
        return []
    texts = {}
    for document in dataset.docs_iter():
        doc_id = str(document.doc_id)
        if doc_id in wanted:
            if doc_id in texts:
                raise ValueError(f"Duplicate source document ID: {doc_id}")
            texts[doc_id] = document.default_text()
    missing = wanted - texts.keys()
    if missing:
        raise ValueError(f"Candidate documents absent from corpus: {sorted(missing)}")
    queries = {str(query.query_id): query for query in dataset.queries_iter()}
    metadata = _read_metadata(source_directory, wanted)
    records = []
    for row in selected.itertuples(index=False):
        if row.qid not in queries:
            raise ValueError(f"Candidate query absent from input: {row.qid}")
        query = queries[row.qid]
        records.append({
            "qid": row.qid, "doc_id": row.docno,
            "bm25_rank": int(row.rank) + 1, "bm25_score": float(row.score),
            "query": retrieval_text(query), "description": reranking_text(query),
            "reranking_field": reranking_field(query),
            "text": texts[row.docno], "metadata": metadata.get(row.docno, {}),
        })
    return records


def write_candidates(records: list[dict], path: Path) -> None:
    """Write one inspectable record per candidate in compressed JSON Lines."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
