"""Standalone stage: candidates JSONL -> semantic group JSONL and summary."""

import argparse
import gzip
import json
from pathlib import Path
import time

from candidates import write_candidates
from semantic_duplicates import group_candidates
from text_embeddings import DocumentEmbedder, MODEL_ID, MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default="/models/multilingual-e5-small")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty")
    with gzip.open(args.candidates, "rt", encoding="utf-8") as stream:
        candidates = [json.loads(line) for line in stream]
    started = time.monotonic()
    groups = group_candidates(candidates, DocumentEmbedder(args.model_path, args.batch_size), args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    write_candidates(groups, args.output / "duplicate-groups.jsonl.gz")
    summary = {"model": MODEL_ID, "revision": MODEL_REVISION, "threshold": args.threshold,
               "candidates": len(candidates), "groups": len(groups),
               "multi_member_groups": sum(len(g["members"]) > 1 for g in groups),
               "seconds": round(time.monotonic() - started, 3),
               "policy": "review groups; preserve all IDs; highest BM25 rank represents group",
               "embedding": "480-token chunks, token-count-weighted mean, normalized"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
