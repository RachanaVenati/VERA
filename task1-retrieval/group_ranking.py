"""Query-aware document ordering after cross-encoder scoring.

No document is removed and scores are never transferred between group members.
MMR uses per-query min-max-normalized relevance and nonnegative document cosine.
Legally incompatible pairs receive no redundancy penalty.
"""

from collections import defaultdict
import math

STRATEGIES = ("cross-encoder", "group-winners", "mmr")


def order_candidates(records, groups=None, similarities=None,
                     strategy="cross-encoder", mmr_lambda=0.8):
    if strategy not in STRATEGIES or not math.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
        raise ValueError("Invalid ranking strategy or MMR lambda.")
    by_query, seen = defaultdict(list), set()
    for record in records:
        key = (str(record["qid"]), str(record["doc_id"]))
        if key in seen or not math.isfinite(record["rerank_score"]):
            raise ValueError("Repeated candidate or invalid relevance score.")
        seen.add(key)
        by_query[key[0]].append(dict(record))
    membership = {}
    if strategy == "group-winners":
        if groups is None:
            raise ValueError("Group-winner ordering requires semantic groups.")
        for index, group in enumerate(groups):
            for member in group["members"]:
                key = (str(group["qid"]), str(member["doc_id"]))
                if key in membership:
                    raise ValueError("Candidate appears in multiple semantic groups.")
                membership[key] = index
        if not seen.issubset(membership):
            raise ValueError("Groups do not cover every scored candidate.")
    pair_scores = {}
    if strategy == "mmr":
        if similarities is None:
            raise ValueError("MMR requires all within-query similarity pairs.")
        for pair in similarities:
            a, b = str(pair["left_id"]), str(pair["right_id"])
            key = (str(pair["qid"]), *sorted((a, b)))
            score = float(pair["cosine"])
            if a == b or key in pair_scores or not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("Invalid or repeated similarity pair.")
            pair_scores[key] = max(0.0, score) if pair["legal_identity_allowed"] else 0.0
    output = []
    for qid in sorted(by_query):
        ordered = sorted(by_query[qid], key=lambda r: (-r["rerank_score"], r["bm25_rank"], str(r["doc_id"])))
        for rank, record in enumerate(ordered, 1):
            record["cross_encoder_rank"] = rank
        if strategy == "group-winners":
            winners, remaining, represented = [], [], set()
            for record in ordered:
                group = membership[(qid, str(record["doc_id"]))]
                (remaining if group in represented else winners).append(record)
                represented.add(group)
            ordered = winners + remaining
        elif strategy == "mmr":
            for index, record in enumerate(ordered):
                for other in ordered[index + 1:]:
                    if (qid, *sorted((str(record["doc_id"]), str(other["doc_id"])))) not in pair_scores:
                        raise ValueError("Missing within-query similarity pair for MMR.")
            minimum = min(r["rerank_score"] for r in ordered)
            span = max(r["rerank_score"] for r in ordered) - minimum
            relevance = {str(r["doc_id"]): (r["rerank_score"] - minimum) / span if span else 1.0 for r in ordered}
            selected = []
            while ordered:
                def objective(record):
                    redundancy = max((pair_scores[(qid, *sorted((str(record["doc_id"]), str(other["doc_id"]))))]
                                      for other in selected), default=0.0)
                    return mmr_lambda * relevance[str(record["doc_id"])] - (1 - mmr_lambda) * redundancy
                # max preserves the original relevance/BM25 tie order.
                winner = max(ordered, key=objective)
                winner["selection_score"] = objective(winner)
                selected.append(winner)
                ordered.remove(winner)
            ordered = selected
        for rank, record in enumerate(ordered, 1):
            record["rerank_rank"] = rank
            record["ranking_strategy"] = strategy
        output.extend(ordered)
    return output
