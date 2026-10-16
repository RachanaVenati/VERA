import gzip
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from event_logging import EventTrace
from submission import merge_rankings, write_run
from cross_encoder import CrossEncoderScorer


class SubmissionTest(unittest.TestCase):
    def test_failed_model_batch_is_logged_before_exception(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.gz"
            trace = EventTrace(path, "model")
            root = trace.emit("1", "input")
            scorer = CrossEncoderScorer.__new__(CrossEncoderScorer)
            scorer.batch_size = 1
            scorer.torch = SimpleNamespace(inference_mode=nullcontext)
            scorer.tokenizer = SimpleNamespace(pad=lambda *a, **k: {}, decode=str)
            def failing_model(**kwargs):
                raise RuntimeError("failed inference")
            scorer.model = failing_model
            scorer.trace_context = {"trace": trace, "qid": "1", "doc_id": "doc",
                                    "description": "travel", "parent": root}
            with self.assertRaisesRegex(RuntimeError, "failed inference"):
                scorer._score_features([{"input_ids": [1,2]}])
            trace.close()
            with gzip.open(path, "rt") as stream:
                events = [json.loads(line) for line in stream]
            self.assertEqual(events[-1]["event_type"], "model_call")
            self.assertEqual(events[-1]["status"], "error")
            self.assertEqual(events[-1]["parent_event_id"], root)
    def bm25(self):
        return pd.DataFrame([["a", "one", 0, 20], ["a", "two", 1, 10],
                             ["a", "three", 2, 8], ["b", "four", 0, 30]],
                            columns=["qid", "docno", "rank", "score"])

    def test_negative_logits_still_precede_bm25_tail(self):
        prefix = [{"qid": "a", "doc_id": "two", "rerank_rank": 1, "rerank_score": -5},
                  {"qid": "a", "doc_id": "one", "rerank_rank": 2, "rerank_score": -9}]
        result = merge_rankings(self.bm25(), prefix)
        self.assertEqual([r["doc_id"] for r in result], ["two", "one", "three", "four"])
        self.assertEqual([r["rank"] for r in result], [1, 2, 3, 1])
        self.assertEqual([r["score"] for r in result[:3]], [1, 0.5, 1/3])

    def test_incomplete_or_unknown_prefix_fails(self):
        for qid, doc_id in [("a", "three"), ("z", "missing")]:
            with self.assertRaises(ValueError):
                merge_rankings(self.bm25(), [{"qid": qid, "doc_id": doc_id, "rerank_rank": 1}])

    def test_bm25_mode_and_output_keep_all_ids(self):
        result = merge_rankings(self.bm25(), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run.txt.gz"
            write_run(result, path)
            with gzip.open(path, "rt") as stream:
                lines = [line.split() for line in stream]
        self.assertEqual([line[2] for line in lines], ["one", "two", "three", "four"])
        self.assertTrue(all(len(line) == 6 for line in lines))

    def test_event_contract_and_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.gz"
            trace = EventTrace(path, "model")
            root = trace.emit("1", "input", inputs={"description": "travel"})
            trace.emit("1", "model_call", root, status="error", error="failed")
            trace.close()
            with gzip.open(path, "rt") as stream:
                events = [json.loads(line) for line in stream]
        self.assertEqual(events[1]["parent_event_id"], root)
        self.assertEqual(events[1]["status"], "error")
        self.assertEqual(events[1]["error"], "failed")
        self.assertEqual(events[1]["model"], "model")
        self.assertEqual(len({e["event_id"] for e in events}), 2)
