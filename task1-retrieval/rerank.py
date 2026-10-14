"""Standalone stage: BM25 candidates + optional groups -> description reranking."""

import argparse
import gzip
import json
from pathlib import Path
import time

from candidates import write_candidates
from cross_encoder import CrossEncoderScorer, MODEL_ID, MODEL_REVISION, DEFAULT_MODEL_PATH
from reranking import rerank_candidates, write_reranked_run


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--groups", type=Path, help="Optional semantic-group artifact to annotate results.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty")
    candidates = read_records(args.candidates)
    groups = read_records(args.groups) if args.groups else None
    started = time.monotonic()
    scorer = CrossEncoderScorer(args.model_path, args.batch_size, args.max_length, args.chunk_overlap)
    results = rerank_candidates(candidates, scorer, groups)
    args.output.mkdir(parents=True, exist_ok=True)
    write_candidates(results, args.output / "reranked-candidates.jsonl.gz")
    write_reranked_run(results, args.output / "run.txt.gz")
    unique_pairs = {(r["description"], r["text"]) for r in candidates}
    summary = {"model": MODEL_ID, "revision": MODEL_REVISION,
               "model_path": args.model_path, "query_field": "description",
               "candidates": len(candidates), "unique_description_text_pairs": len(unique_pairs),
               "score_reuses": len(candidates) - len(unique_pairs),
               "max_length": args.max_length, "chunk_overlap": args.chunk_overlap,
               "aggregation": "maximum chunk relevance logit", "batch_size": args.batch_size,
               "semantic_groups": "annotations only; scores reused for identical text only",
               "output_scope": "candidate set only; original BM25 tail is not merged",
               "seconds": round(time.monotonic() - started, 3)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
