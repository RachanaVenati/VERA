"""Evaluate pool coverage, scored coverage, final ranking, and runtime separately."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def load_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def pool_diagnostics(run, scores, qrels, candidate_k):
    """Recover original BM25 ranks from a permuted scored prefix and its tail."""
    original, final, scored = defaultdict(dict), defaultdict(dict), defaultdict(set)
    positions = {}
    for row in scores:
        qid, doc_id = str(row["qid"]), str(row["doc_id"])
        if (qid, doc_id) in positions or not 1 <= row["bm25_rank"] <= candidate_k:
            raise ValueError("Repeated scored document or score outside requested pool.")
        positions[(qid, doc_id)] = row["bm25_rank"]
        scored[qid].add(doc_id)
    for row in run:
        qid, doc_id = str(row["qid"]), str(row["doc_id"])
        if doc_id in final[qid]:
            raise ValueError("Repeated final document ID.")
        final[qid][doc_id] = row["rank"]
        original[qid][doc_id] = positions.get((qid, doc_id), row["rank"])
    if any(doc not in final[qid] for qid, doc in positions):
        raise ValueError("Scored document missing from final ranking.")
    for qid in final:
        expected = list(range(1, len(final[qid]) + 1))
        if sorted(final[qid].values()) != expected or sorted(original[qid].values()) != expected:
            raise ValueError("Invalid final ranking or original BM25 prefix reconstruction.")
    relevant = defaultdict(set)
    for row in qrels:
        if float(row["grade"]) > 0:
            relevant[str(row["qid"])].add(str(row["doc_id"]))
    per_query = []
    for qid in sorted(relevant):
        targets = relevant[qid]
        pool_hits = {d for d in targets if original[qid].get(d, candidate_k + 1) <= candidate_k}
        final_hits = {d for d in targets if final[qid].get(d, 1001) <= 1000}
        per_query.append({"qid": qid, "candidate_recall": len(pool_hits) / len(targets),
                          "scored_recall": len(targets & scored[qid]) / len(targets),
                          "recall_at_1000": len(final_hits) / len(targets),
                          "scored_documents": len(scored[qid]),
                          "targets": [{"doc_id": d, "bm25_rank": original[qid].get(d),
                                       "final_rank": final[qid].get(d), "scored": d in scored[qid]}
                                      for d in sorted(targets)]})
    def mean(key):
        return sum(row[key] for row in per_query) / len(per_query) if per_query else None
    return {"candidate_recall": mean("candidate_recall"), "scored_recall": mean("scored_recall"),
            "Recall@1000": mean("recall_at_1000"), "scored_documents": len(scores), "per_query": per_query}


def evaluate(source, truths, output):
    from evaluate_submission import run_tira_evaluate
    summary = json.loads((source / "summary.json").read_text())
    if (summary["mode"] != "rerank" or summary.get("ranking_strategy") != "cross-encoder"
            or summary.get("semantic_audit", {}).get("mode") != "off"):
        raise ValueError("Pool-size experiment requires plain cross-encoder reranking with semantic auditing off.")
    run = []
    with gzip.open(source / "run.txt.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            run.append({"qid": fields[0], "doc_id": fields[2], "rank": int(fields[3])})
    qrels = []
    for line in (truths / "qrels.txt").read_text().splitlines():
        qid, _, doc_id, grade = line.split()
        qrels.append({"qid": qid, "doc_id": doc_id, "grade": grade})
    report = {"dataset": truths.name, "candidate_k": summary["candidate_k"],
              "metrics": run_tira_evaluate(source, truths, "retrieval"),
              "coverage": pool_diagnostics(run, load_records(source / "reranking-scores.jsonl.gz"),
                                           qrels, summary["candidate_k"]),
              "source_summary": summary,
              "protocol": "Fresh inference; same 4 CPU cores, cross-encoder, fields and 240s query budget; semantic audit off."}
    if output.exists():
        raise ValueError("Evaluation output already exists.")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.source, args.truths, args.output)


if __name__ == "__main__":
    main()
