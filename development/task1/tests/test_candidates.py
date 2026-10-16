import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from candidates import prepare_candidates, select_candidates, write_candidates
from test_baseline import persisted_dataset


def run(rows):
    return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])


class CandidatesTest(unittest.TestCase):
    def test_selection_is_per_query_and_uses_rank_not_input_order(self):
        results = run([
            ["a", "d2", 1, 5.0], ["b", "d2", 0, 8.0],
            ["a", "d1", 0, 9.0], ["b", "d1", 1, 4.0],
        ])
        selected = select_candidates(results, 1)
        self.assertEqual(list(zip(selected.qid, selected.docno)), [("a", "d1"), ("b", "d2")])
        self.assertEqual(len(results), 4)

    def test_repeated_ids_do_not_consume_candidate_slots(self):
        selected = select_candidates(run([
            ["a", "d1", 0, 9.0], ["a", "d1", 1, 8.0], ["a", "d2", 2, 7.0],
        ]), 2)
        self.assertEqual(list(selected.docno), ["d1", "d2"])

    def test_text_description_metadata_and_json_output(self):
        queries = [{"qid": "1", "query": "travel", "original_query": {
            "language": "en", "description": "travel reimbursement rules"}}]
        docs = [{"doc_id": "d1", "text": "travel document", "decision_date": "01.01.2020"},
                {"doc_id": "d2", "text": "unused"}]
        with persisted_dataset(queries, docs) as dataset:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "documents.jsonl").write_text(
                    "".join(json.dumps(doc) + "\n" for doc in docs))
                records = prepare_candidates(dataset, run([["1", "d1", 0, 2.5]]), 50, directory)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["text"], "travel document")
                self.assertEqual(records[0]["query"], "travel")
                self.assertEqual(records[0]["description"], "travel reimbursement rules")
                self.assertEqual(records[0]["metadata"]["decision_date"], "01.01.2020")
                self.assertEqual(records[0]["bm25_rank"], 1)
                output = directory / "candidates.jsonl.gz"
                write_candidates(records, output)
                with gzip.open(output, "rt") as stream:
                    self.assertEqual([json.loads(line) for line in stream], records)

    def test_missing_document_fails(self):
        with persisted_dataset(
            [{"qid": "1", "query": "travel"}], [{"doc_id": "d1", "text": "travel"}],
        ) as dataset:
            with self.assertRaisesRegex(ValueError, "absent from corpus"):
                prepare_candidates(dataset, run([["1", "missing", 0, 1.0]]))

    def test_empty_run_and_invalid_count(self):
        self.assertTrue(select_candidates(run([]), 50).empty)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            select_candidates(run([]), 0)
