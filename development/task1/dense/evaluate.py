"""Compare independent pure-dense and BM25 runs after retrieval completes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_submission import run_tira_evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--truths', type=Path, required=True)
    args = parser.parse_args()
    report = {'dataset': args.truths.name}
    for system in ('bm25','dense'):
        path = args.results / system
        report[system] = {'metrics': run_tira_evaluate(path, args.truths, 'retrieval', allow_small_retrieval_subset=True),
                          'summary': json.loads((path/'summary.json').read_text())}
    (args.results/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
