import gzip
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from compare_group_rankings import compare, recall_at_1000, restore_bm25_order
from test_semantic_pipeline import load_pipeline


class RankingComparisonTest(unittest.TestCase):
    def test_restore_original_prefix_and_untouched_tail(self):
        run = [{"qid": "1", "doc_id": doc, "rank": rank, "score": 1 / rank}
               for rank, doc in enumerate(["b", "a", "c"], 1)]
        scores = [{"qid": "1", "doc_id": "a", "bm25_rank": 1},
                  {"qid": "1", "doc_id": "b", "bm25_rank": 2}]
        frame = restore_bm25_order(run, scores)
        self.assertEqual(list(frame.sort_values("rank").docno), ["a", "b", "c"])

    def test_full_comparison_reuses_scores_and_preserves_tail(self):
        pipeline = load_pipeline()
        evaluator = ModuleType("evaluate_submission")
        # Test orchestration, not metric arithmetic; the workflow uses TIRA's evaluator.
        evaluator.run_tira_evaluate = lambda *args, **kwargs: {"nDCG@10": 0.5}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, truths, output = root / "source", root / "truths", root / "comparison"
            source.mkdir()
            truths.mkdir()
            (truths / "qrels.txt").write_text("1 0 a 1\n1 0 b 1\n2 0 z 1\n")
            (source / "summary.json").write_text(json.dumps({"semantic_audit": {"mode": "audit"},
                                                            "ranking_strategy": "cross-encoder"}))
            records = [dict(qid="1", doc_id="a", rerank_score=10, bm25_rank=1),
                       dict(qid="1", doc_id="b", rerank_score=9.5, bm25_rank=2),
                       dict(qid="1", doc_id="c", rerank_score=9.49, bm25_rank=3)]
            from test_group_ranking import groups, pairs
            for name, rows in [("reranking-scores", records), ("semantic-groups", groups()),
                               ("semantic-similarities", pairs())]:
                with gzip.open(source / f"{name}.jsonl.gz", "wt") as stream:
                    stream.writelines(json.dumps(row) + "\n" for row in rows)
            with gzip.open(source / "run.txt.gz", "wt") as stream:
                for rank, doc in enumerate(["a", "b", "c", "tail"], 1):
                    stream.write(f"1 Q0 {doc} {rank} {1 / rank} vera\n")
            with patch.dict("sys.modules", {"submission": pipeline, "evaluate_submission": evaluator}):
                result = compare(source, truths, output)
            self.assertEqual(len(result["policies"]), 5)
            self.assertEqual(result["selected_on_calibration"], "cross-encoder")
            self.assertTrue((output / "comparison.json").is_file())
            for name in result["policies"]:
                with gzip.open(output / name / "run.txt.gz", "rt") as stream:
                    docs = [line.split()[2] for line in stream]
                self.assertEqual(set(docs), {"a", "b", "c", "tail"})
                self.assertEqual(docs[-1], "tail")

    def test_recall_uses_all_positive_judgments_including_missing_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qrels.txt"
            path.write_text("1 0 a 1\n1 0 b 2\n1 0 c 0\n2 0 missing 1\n")
            score = recall_at_1000([{"qid": "1", "doc_id": "a", "rank": 1}], path)
            self.assertEqual(score, 0.25)


if __name__ == "__main__":
    unittest.main()
