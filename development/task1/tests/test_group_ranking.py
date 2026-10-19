import unittest

from group_ranking import order_candidates


def candidate(doc_id, score, bm25_rank, qid="1"):
    return {"qid": qid, "doc_id": doc_id, "rerank_score": score, "bm25_rank": bm25_rank}


def groups():
    return [{"qid": "1", "representative_id": "a", "members": [{"doc_id": "a"}, {"doc_id": "b"}]},
            {"qid": "1", "representative_id": "c", "members": [{"doc_id": "c"}]}]


def pairs(legal=True):
    return [{"qid": "1", "left_id": a, "right_id": b, "cosine": cosine,
             "legal_identity_allowed": legal if (a, b) == ("a", "b") else True}
            for a, b, cosine in [("a", "b", 1.0), ("a", "c", 0.0), ("b", "c", 0.2)]]


class GroupRankingTest(unittest.TestCase):
    def test_winner_is_best_description_score_not_bm25_representative(self):
        records = [candidate("a", 9, 1), candidate("b", 10, 2), candidate("c", 6, 3)]
        result = order_candidates(records, groups(), strategy="group-winners")
        self.assertEqual([r["doc_id"] for r in result], ["b", "c", "a"])
        self.assertEqual([r["rerank_score"] for r in result], [10, 6, 9])
        self.assertEqual([r["rerank_rank"] for r in result], [1, 2, 3])
        self.assertNotIn("rerank_rank", records[0])

    def test_mmr_can_promote_less_redundant_document(self):
        records = [candidate("a", 10, 1), candidate("b", 9.5, 2), candidate("c", 9.49, 3)]
        result = order_candidates(records, similarities=pairs(), strategy="mmr")
        self.assertEqual([r["doc_id"] for r in result], ["a", "c", "b"])
        self.assertEqual([r["cross_encoder_rank"] for r in result], [1, 3, 2])

    def test_lambda_one_is_plain_relevance_order(self):
        records = [candidate("b", 9.5, 2), candidate("a", 10, 1), candidate("c", 9.49, 3)]
        baseline = order_candidates(records)
        mmr = order_candidates(records, similarities=pairs(), strategy="mmr", mmr_lambda=1)
        self.assertEqual([r["doc_id"] for r in baseline], [r["doc_id"] for r in mmr])

    def test_different_legal_records_are_not_penalized(self):
        records = [candidate("a", 10, 1), candidate("b", 9.5, 2), candidate("c", 9.49, 3)]
        result = order_candidates(records, similarities=pairs(False), strategy="mmr")
        self.assertEqual([r["doc_id"] for r in result], ["a", "b", "c"])

    def test_ties_deterministic_and_queries_separate(self):
        records = [candidate("b", 1, 2), candidate("a", 1, 1), candidate("z", 7, 1, "2")]
        result = order_candidates(records)
        self.assertEqual([r["doc_id"] for r in result], ["a", "b", "z"])
        self.assertEqual([r["rerank_rank"] for r in result], [1, 2, 1])

    def test_scored_subset_does_not_require_unscored_group_members(self):
        result = order_candidates([candidate("b", 9, 2)], groups(), strategy="group-winners")
        self.assertEqual([r["doc_id"] for r in result], ["b"])

    def test_missing_or_overlapping_groups_and_similarity_pairs_fail(self):
        record = candidate("a", 1, 1)
        for mapping in (None, [], groups() + groups()):
            with self.assertRaises(ValueError):
                order_candidates([record], mapping, strategy="group-winners")
        with self.assertRaisesRegex(ValueError, "Missing"):
            order_candidates([record, candidate("b", 2, 2)], similarities=[], strategy="mmr")

    def test_invalid_scores_or_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            order_candidates([candidate("a", float("nan"), 1)])
        with self.assertRaises(ValueError):
            order_candidates([candidate("a", 1, 1)] * 2)
        with self.assertRaises(ValueError):
            order_candidates([], strategy="unknown")


if __name__ == "__main__":
    unittest.main()
