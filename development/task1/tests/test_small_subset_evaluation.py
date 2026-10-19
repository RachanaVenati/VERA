"""Regression: real TIRA evaluation of intentional one-query/one-qrel splits."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("trectools"), "Run in the submission image with trectools")
class SmallSubsetEvaluationTest(unittest.TestCase):
    def test_small_split_keeps_metric_and_full_run_defaults(self):
        from evaluate_submission import run_tira_evaluate, TASK_CONFIGS
        import click
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = root / "predictions", root / "truths"
            predictions.mkdir()
            truths.mkdir()
            (predictions / "run.txt").write_text("q Q0 relevant 1 1.0 test\n")
            (truths / "qrels.txt").write_text("q 0 relevant 1\n")
            with self.assertRaises(click.ClickException):
                run_tira_evaluate(predictions, truths, "retrieval")
            measures = run_tira_evaluate(predictions, truths, "retrieval", allow_small_retrieval_subset=True)
            self.assertAlmostEqual(measures["nDCG@10"], 1.0)
            self.assertNotIn("format_configuration", TASK_CONFIGS["retrieval"])
            with self.assertRaises(click.ClickException):
                run_tira_evaluate(predictions, truths, "retrieval")

    def test_malformed_input_still_fails(self):
        from evaluate_submission import run_tira_evaluate
        import click
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = root / "predictions", root / "truths"
            predictions.mkdir()
            truths.mkdir()
            (predictions / "run.txt").write_text("not a valid trec row\n")
            (truths / "qrels.txt").write_text("q 0 relevant 1\n")
            with self.assertRaises(click.ClickException):
                run_tira_evaluate(predictions, truths, "retrieval", allow_small_retrieval_subset=True)

    def test_duplicate_documents_still_fail(self):
        from evaluate_submission import run_tira_evaluate
        import click
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = root / "predictions", root / "truths"
            predictions.mkdir()
            truths.mkdir()
            (predictions / "run.txt").write_text("q Q0 relevant 1 1.0 test\nq Q0 relevant 2 0.5 test\n")
            (truths / "qrels.txt").write_text("q 0 relevant 1\n")
            with self.assertRaises(click.ClickException):
                run_tira_evaluate(predictions, truths, "retrieval", allow_small_retrieval_subset=True)


if __name__ == "__main__":
    unittest.main()
