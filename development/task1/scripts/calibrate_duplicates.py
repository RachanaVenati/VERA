"""Report semantic thresholds from manually labeled review-pairs.csv."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "task1-retrieval"))

from semantic_review import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.90, 0.93, 0.95, 0.97, 0.98, 0.99])
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--min-predictions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    with args.reviews.open(encoding="utf-8", newline="") as stream:
        report = calibrate(list(csv.DictReader(stream)), args.thresholds,
                           args.target_precision, args.min_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
