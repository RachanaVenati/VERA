"""Compare ordering policies using one frozen audit run, without new inference.

The TIRA evaluator supplies nDCG@10. Selection uses calibration queries only;
the held-out split is reported separately. All policies keep the full ID set.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from group_ranking import order_candidates


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_run(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [{"qid": fields[0], "doc_id": fields[2], "rank": int(fields[3]), "score": float(fields[4])}
                for line in stream if (fields := line.split())]


def restore_bm25_order(run, scores):
    """Undo scored-prefix permutations; unchanged tail ranks already match BM25."""
    positions = {(str(r["qid"]), str(r["doc_id"])): r["bm25_rank"] for r in scores}
    rows = [{"qid": r["qid"], "docno": r["doc_id"],
             "rank": positions.get((r["qid"], r["doc_id"]), r["rank"]) - 1,
             "score": r["score"]} for r in run]
    frame = pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
    for qid, query in frame.groupby("qid"):
        if sorted(query["rank"]) != list(range(len(query))) or query.docno.duplicated().any():
            raise ValueError(f"Cannot reconstruct a complete BM25 ordering for {qid}.")
    return frame


def recall_at_1000(run, qrels):
    relevant, retrieved = {}, {}
    for line in qrels.read_text().splitlines():
        qid, _, doc_id, grade = line.split()
        if float(grade) > 0:
            relevant.setdefault(qid, set()).add(doc_id)
    for row in run:
        if row["rank"] <= 1000:
            retrieved.setdefault(row["qid"], set()).add(row["doc_id"])
    values = [len(ids & retrieved.get(qid, set())) / len(ids) for qid, ids in relevant.items()]
    return sum(values) / len(values) if values else None


def split_for_query(qid):
    return "validation" if int(hashlib.sha256(str(qid).encode()).hexdigest(), 16) % 5 == 0 else "calibration"


def compare(source, truths, output):
    # Imported only inside the container; ordering unit tests need no JVM.
    from submission import merge_rankings, write_run
    from evaluate_submission import run_tira_evaluate

    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be empty.")
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("semantic_audit", {}).get("mode") != "audit":
        raise ValueError("Comparison requires a completed semantic-audit run.")
    if summary.get("ranking_strategy", "cross-encoder") != "cross-encoder":
        raise ValueError("Source must use the plain cross-encoder ordering.")
    run = read_run(source / "run.txt.gz")
    scores = read_jsonl(source / "reranking-scores.jsonl.gz")
    groups = read_jsonl(source / "semantic-groups.jsonl.gz")
    similarities = read_jsonl(source / "semantic-similarities.jsonl.gz")
    bm25 = restore_bm25_order(run, scores)
    expected = {(r["qid"], r["doc_id"]) for r in run}
    output.mkdir(parents=True, exist_ok=True)
    qrels_lines = (truths / "qrels.txt").read_text().splitlines()
    splits = {}
    for split in ("calibration", "validation"):
        path = output / "truths" / split
        path.mkdir(parents=True)
        lines = [line for line in qrels_lines if split_for_query(line.split()[0]) == split]
        (path / "qrels.txt").write_text("\n".join(lines) + "\n")
        splits[split] = (path, {line.split()[0] for line in lines})
    reports = {}
    policies = [("cross-encoder", "cross-encoder", 0.8), ("group-winners", "group-winners", 0.8)]
    policies += [(f"mmr-{weight}", "mmr", weight) for weight in (0.7, 0.8, 0.9)]
    for name, strategy, weight in policies:
        started = time.monotonic()
        ranked = order_candidates(scores, groups, similarities, strategy, weight)
        final = merge_rankings(bm25, ranked)
        seconds = time.monotonic() - started
        if {(r["qid"], r["doc_id"]) for r in final} != expected or len(final) != len(run):
            raise ValueError("Policy changed the original document ID set.")
        if name == "cross-encoder" and [(r["qid"], r["doc_id"], r["rank"]) for r in final] != [
                (r["qid"], r["doc_id"], r["rank"]) for r in run]:
            raise ValueError("Reconstructed cross-encoder baseline differs from the source run.")
        directory = output / name
        directory.mkdir()
        write_run(final, directory / "run.txt.gz")
        report = {"all": run_tira_evaluate(directory, truths, "retrieval"),
                  "Recall@1000": recall_at_1000(final, truths / "qrels.txt"),
                  "reordering_seconds": seconds, "strategy": strategy,
                  "mmr_lambda": weight if strategy == "mmr" else None}
        for split, (qrels_dir, qids) in splits.items():
            subset = [r for r in final if r["qid"] in qids]
            target = directory / split
            target.mkdir()
            write_run(subset, target / "run.txt.gz")
            report[split] = run_tira_evaluate(target, qrels_dir, "retrieval",
                                            allow_small_retrieval_subset=True) if qids else None
        reports[name] = report
    base = reports["cross-encoder"]["all"]["nDCG@10"]
    for report in reports.values():
        report["delta_nDCG@10"] = report["all"]["nDCG@10"] - base
    eligible = [name for name, report in reports.items() if report["calibration"] is not None]
    selected = max(eligible, key=lambda name: reports[name]["calibration"]["nDCG@10"]) if eligible else None
    result = {"policies": reports, "selected_on_calibration": selected,
              "selected_validation": reports[selected]["validation"] if selected else None,
              "source_summary": summary,
              "split_query_counts": {key: len(value[1]) for key, value in splits.items()},
              "protocol": "One inference run; fixed scores and candidate IDs; no automatic deployment. Sample splits are small."}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    table = ["| Policy | nDCG@10 (all) | Delta | Calibration | Validation | Recall@1000 |",
             "|---|---:|---:|---:|---:|---:|"]
    def value(report, split):
        return f"{report[split]['nDCG@10']:.6f}" if report[split] else "n/a"
    for name, report in reports.items():
        recall = report["Recall@1000"]
        table.append(f"| {name} | {report['all']['nDCG@10']:.6f} | {report['delta_nDCG@10']:+.6f} | "
                     f"{value(report, 'calibration')} | {value(report, 'validation')} | {recall if recall is not None else 'n/a'} |")
    markdown = "\n".join(table) + f"\n\nSelected on calibration only: {selected}. No automatic deployment.\n"
    (output / "comparison.md").write_text(markdown)
    print(markdown, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.source, args.truths, args.output)


if __name__ == "__main__":
    main()
