import unittest

from evaluate_candidate_pool import pool_diagnostics


class CandidatePoolTest(unittest.TestCase):
    def test_candidate_coverage_differs_from_completed_scoring(self):
        run = [{"qid": "q", "doc_id": d, "rank": rank}
               for rank, d in enumerate(["b", "a", "c", "d"], 1)]
        scores = [{"qid": "q", "doc_id": "a", "bm25_rank": 1},
                  {"qid": "q", "doc_id": "b", "bm25_rank": 2}]
        qrels = [{"qid": "q", "doc_id": "c", "grade": 1}]
        report = pool_diagnostics(run, scores, qrels, 3)
        self.assertEqual(report["candidate_recall"], 1)
        self.assertEqual(report["scored_recall"], 0)
        self.assertEqual(report["per_query"][0]["targets"][0]["bm25_rank"], 3)

    def test_new_target_at_rank_96_enters_only_larger_pool(self):
        run = [{"qid": "q", "doc_id": str(rank), "rank": rank} for rank in range(1, 101)]
        qrels = [{"qid": "q", "doc_id": "96", "grade": 1}]
        self.assertEqual(pool_diagnostics(run, [], qrels, 50)["candidate_recall"], 0)
        self.assertEqual(pool_diagnostics(run, [], qrels, 100)["candidate_recall"], 1)

    def test_missing_query_counts_as_zero_recall(self):
        qrels = [{"qid": "missing", "doc_id": "target", "grade": 1}]
        self.assertEqual(pool_diagnostics([], [], qrels, 100)["candidate_recall"], 0)

    def test_invalid_prefix_reconstruction_is_rejected(self):
        run = [{"qid": "q", "doc_id": "a", "rank": 1}]
        with self.assertRaises(ValueError):
            pool_diagnostics(run, [{"qid": "q", "doc_id": "a", "bm25_rank": 2}], [], 100)


if __name__ == "__main__":
    unittest.main()
