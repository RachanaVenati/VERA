"""Audit final TREC rankings and UniAgent traces without using relevance truths."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

FIELDS = {"case_id", "event_id", "parent_event_id", "timestamp", "event_type",
          "model", "tool", "input", "output", "status", "error"}


def read_run(path, strict=True):
    rows = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            assert len(fields) == 6, "Expected six TREC fields"
            qid, _, doc_id, rank, score, _ = fields
            assert math.isfinite(float(score))
            rows[qid].append((doc_id, int(rank), float(score)))
    for qid, results in rows.items():
        assert len(results) <= 1000
        assert len({r[0] for r in results}) == len(results)
        if strict:
            assert [r[1] for r in results] == list(range(1, len(results)+1))
            assert all(results[i][2] > results[i+1][2] for i in range(len(results)-1))
    return rows


def audit(output, dataset, baseline=None):
    queries = {str(q["qid"]): q for q in map(json.loads,
               (dataset/"queries.jsonl").read_text().splitlines())}
    run = read_run(output/"run.txt.gz")
    assert set(run) <= queries.keys()
    summary = json.loads((output/"summary.json").read_text())
    ids, last, counts, decisions = {}, {}, Counter(), {}
    with gzip.open(output/"run-trace.jsonl.log.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            assert FIELDS <= event.keys()
            qid = event["case_id"]
            assert qid in queries and event["event_id"] not in ids
            assert event["model"] == summary["model"]
            assert event["status"] == "ok", event.get("error")
            if event["parent_event_id"]:
                assert ids[event["parent_event_id"]] == qid
            ids[event["event_id"]] = qid
            last[qid] = event["event_type"]
            counts[event["event_type"]] += 1
            if event["event_type"] == "model_call":
                assert event["input"]["description"].strip() == queries[qid]["original_query"]["description"].strip()
                pairs = event["input"]["pairs"]
                logits = event["output"]["chunk_relevance_logits"]
                assert len(pairs) == len(logits)
                assert all(len(pair["input_ids"]) <= 512 for pair in pairs)
                assert all(math.isfinite(logit) for logit in logits)
            if event["event_type"] == "decision":
                answer = event["output"]["answer"]
                assert [(r["doc_id"], r["rank"]) for r in answer] == [(r[0],r[1]) for r in run.get(qid, [])]
                decisions[qid] = True
    assert set(last) == queries.keys() == decisions.keys()
    assert all(kind == "decision" for kind in last.values())
    if baseline:
        original = read_run(baseline, strict=False)
        assert {q: {r[0] for r in rows} for q,rows in run.items()} == {
            q: {r[0] for r in rows} for q,rows in original.items()}
    return {"queries": len(queries), "results": sum(map(len,run.values())),
            "events": dict(counts), "seconds": summary["seconds"],
            "max_case_reranking_seconds": max(summary["case_reranking_seconds"].values(), default=0),
            "budget_limited_queries": summary["budget_limited_queries"], "audit": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.dataset, args.baseline), indent=2))
