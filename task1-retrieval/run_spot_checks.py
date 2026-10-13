"""Run and evaluate all retrieval spot checks inside the submission container."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_submission import run_tira_evaluate

DATASETS = (
    "retrieval-de-spot-check",
    "retrieval-en-spot-check",
    "retrieval-hessian-law-de-spot-check",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wmodel", default="BM25")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty; choose a new directory for each experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in DATASETS:
        dataset = args.datasets / name
        output = args.output / name
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("baseline.py")),
             "--dataset", str(dataset), "--output", str(output),
             "--wmodel", args.wmodel],
            check=True,
        )
        reports[name] = run_tira_evaluate(output, dataset, "retrieval")
        print(f"{name}: {json.dumps(reports[name])}", flush=True)
        (output / "evaluation.json").write_text(
            json.dumps(reports[name], indent=2) + "\n", encoding="utf-8"
        )
    (args.output / "summary.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
