"""Evaluate only after retrieval; no judgments are used for indexing."""
import argparse,gzip,json,tempfile
from pathlib import Path
from trectools import TrecRun,TrecQrel,TrecEval

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--qrels',type=Path,required=True)
args=parser.parse_args()
with tempfile.TemporaryDirectory() as d:
    run=Path(d)/'run.txt'
    with gzip.open(args.results/'run.txt.gz','rt') as stream:run.write_text(stream.read())
    evaluator=TrecEval(TrecRun(str(run)),TrecQrel(str(args.qrels)))
    report={'nDCG@10':evaluator.get_ndcg(depth=10),'Recall@100':evaluator.get_recall(depth=100),
            'Recall@1000':evaluator.get_recall(depth=1000)}
(args.results/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
