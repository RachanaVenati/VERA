"""Group semantically similar candidates without deleting document IDs.

The highest-ranked BM25 candidate represents each group. A candidate must
meet the threshold against that representative, not merely another member.
Groups are suggestions for inspection, not confirmed identical legal meaning.
"""

import numpy as np


def _same_legal_record(left: dict, right: dict) -> bool:
    """Ordinary pages are unrestricted; legal records need matching identity."""
    a, b = left.get("metadata", {}), right.get("metadata", {})
    legal = any(key in a or key in b for key in ("ecli", "decision_date", "file_number"))
    if not legal:
        return True
    if a.get("ecli") and b.get("ecli"):
        return a["ecli"] == b["ecli"]
    keys = ("court", "file_number", "decision_date")
    return all(a.get(key) and a[key] == b.get(key) for key in keys)


def group_candidates(records: list[dict], embedder, threshold: float = 0.95) -> list[dict]:
    """Return per-query representative/member mappings and cosine scores.

Text embeddings are reused across queries. All originals remain in the
candidate artifact and every original appears in exactly one output group.
Dates never choose the representative. Metadata guards restrict legal grouping.
"""
    if not 0 < threshold <= 1:
        raise ValueError("Cosine threshold must be in (0, 1].")
    if not records:
        return []
    seen = set()
    for record in records:
        key = (record["qid"], record["doc_id"])
        if key in seen:
            raise ValueError(f"Repeated candidate ID within query: {key}")
        seen.add(key)
    texts = list(dict.fromkeys(record["text"] for record in records))
    vectors = np.asarray(embedder.encode(texts), dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid vectors.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector.")
    vectors = vectors / norms[:, None]
    lookup = {text: vectors[index] for index, text in enumerate(texts)}
    by_query = {}
    for record in sorted(records, key=lambda r: (r["qid"], r["bm25_rank"], r["doc_id"])):
        by_query.setdefault(record["qid"], []).append(record)
    output = []
    for qid, candidates in by_query.items():
        representatives = []
        groups = []
        for candidate in candidates:
            matches = []
            for index, representative in enumerate(representatives):
                score = float(np.clip(lookup[candidate["text"]] @ lookup[representative["text"]], -1, 1))
                if score >= threshold and _same_legal_record(candidate, representative):
                    matches.append((score, index))
            if matches:
                # Highest similarity wins; earlier BM25 representative breaks ties.
                score, index = max(matches, key=lambda match: (match[0], -match[1]))
            else:
                score, index = 1.0, len(groups)
                representatives.append(candidate)
                groups.append({"qid": qid, "representative_id": candidate["doc_id"],
                               "threshold": threshold, "members": []})
            groups[index]["members"].append({
                "doc_id": candidate["doc_id"], "bm25_rank": candidate["bm25_rank"],
                "cosine_to_representative": score,
            })
        output.extend(groups)
    return output
