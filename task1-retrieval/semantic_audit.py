"""Optional post-BM25 audit. Suggest redundancy groups; preserve every ID."""

import gzip
import json
import time

from semantic_review import audit_candidates, write_review
from text_embeddings import MODEL_ID, MODEL_REVISION


def run_semantic_audit(candidates, embedder, output, trace, parents,
                       threshold=0.95, review_floor=0.90):
    started = time.monotonic()
    similarities = []
    groups, pairs = audit_candidates(candidates, embedder, threshold, review_floor, similarities)
    with gzip.open(output / "semantic-similarities.jsonl.gz", "wt", encoding="utf-8") as stream:
        for pair in similarities:
            stream.write(json.dumps(pair, allow_nan=False) + "\n")
    with gzip.open(output / "semantic-groups.jsonl.gz", "wt", encoding="utf-8") as stream:
        for group in groups:
            stream.write(json.dumps(group, ensure_ascii=False, allow_nan=False) + "\n")
    write_review(pairs, output / "semantic-review-pairs.csv")
    for qid in parents:
        trace.emit(qid, "observation", parents[qid], output={
            "stage": "semantic_audit", "model": MODEL_ID, "revision": MODEL_REVISION,
            "threshold": threshold, "review_floor": review_floor,
            "groups": [g for g in groups if str(g["qid"]) == qid],
            "embedding_sources": {r["doc_id"]: getattr(embedder, "source_events", {}).get(r["text"])
                                  for r in candidates if str(r["qid"]) == qid},
            "review_pairs": sum(str(p["qid"]) == qid for p in pairs),
            "policy": "diagnostic only; no ID removal or semantic score reuse"}, model=MODEL_ID)
    return groups, {"mode": "audit", "model": MODEL_ID, "revision": MODEL_REVISION,
                    "threshold": threshold, "review_floor": review_floor,
                    "groups": len(groups), "review_pairs": len(pairs),
                    "multi_member_groups": sum(len(g["members"]) > 1 for g in groups),
                    "seconds": time.monotonic() - started,
                    "policy": "preserve all IDs; score each distinct description/text pair"}, similarities
