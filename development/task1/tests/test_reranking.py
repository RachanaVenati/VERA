import gzip
import tempfile
import unittest
from pathlib import Path

from cross_encoder import CrossEncoderScorer, pair_chunks
from cross_encoder import RerankingBudgetExceeded
from reranking import rerank_candidates, write_reranked_run


def candidate(doc_id, rank, text, description="expense rules", qid="1"):
    return {"qid": qid, "doc_id": doc_id, "bm25_rank": rank,
            "bm25_score": 10-rank, "query": "unused short query", "description": description,
            "text": text, "metadata": {}}


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, description, text):
        self.calls.append((description, text))
        return {"score": float(text), "chunk_count": 1, "best_chunk_index": 0}


class FakeTokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def encode(self, text, **kwargs):
        return [int(value) for value in text.split()]

    def num_special_tokens_to_add(self, pair):
        return 3

    def prepare_for_model(self, ids, pair_ids, **kwargs):
        return {"input_ids": [-1]+ids+[-2]+pair_ids+[-3],
                "document_ids": pair_ids}


class RerankingTest(unittest.TestCase):
    def test_time_budget_retains_contiguous_prefix_and_continues_next_query(self):
        class BudgetScorer(FakeScorer):
            def score(self, description, text):
                if text == "budget":
                    raise RerankingBudgetExceeded()
                return super().score(description, text)
        records = [candidate("a", 1, "1"), candidate("b", 2, "budget"),
                   candidate("c", 3, "9"), candidate("d", 1, "2", qid="2")]
        result = rerank_candidates(records, BudgetScorer())
        self.assertEqual([r["doc_id"] for r in result], ["a", "d"])

    def test_expired_budget_stops_before_model_batch(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        scorer = CrossEncoderScorer.__new__(CrossEncoderScorer)
        scorer.torch = SimpleNamespace(inference_mode=nullcontext)
        scorer.batch_size, scorer.deadline = 1, 0
        with self.assertRaises(RerankingBudgetExceeded):
            scorer._score_features([{}])
    def test_description_scores_reorder_candidates_and_preserve_input(self):
        records = [candidate("a", 1, "1"), candidate("b", 2, "9")]
        scorer = FakeScorer()
        result = rerank_candidates(records, scorer)
        self.assertEqual([r["doc_id"] for r in result], ["b", "a"])
        self.assertEqual([r["rerank_rank"] for r in result], [1, 2])
        self.assertEqual(scorer.calls, [("expense rules", "1"), ("expense rules", "9")])
        self.assertNotIn("rerank_score", records[0])

    def test_semantic_members_scored_individually_exact_text_reuses_score(self):
        records = [candidate("a", 1, "1"), candidate("b", 2, "9"), candidate("c", 3, "1")]
        groups = [{"qid": "1", "representative_id": "a",
                   "members": [{"doc_id": doc_id} for doc_id in ("a", "b", "c")]}]
        scorer = FakeScorer()
        results = rerank_candidates(records, scorer, groups)
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual({r["doc_id"] for r in results}, {"a", "b", "c"})
        by_id = {r["doc_id"]: r for r in results}
        self.assertEqual(by_id["c"]["score_source_doc_id"], "a")
        self.assertEqual(by_id["b"]["rerank_score"], 9)

    def test_same_text_different_descriptions_not_reused_and_ranks_per_query(self):
        scorer = FakeScorer()
        results = rerank_candidates([candidate("a", 1, "1"),
                                     candidate("a", 1, "1", "different question", "2")], scorer)
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual([r["rerank_rank"] for r in results], [1, 1])

    def test_missing_description_and_incomplete_groups_fail(self):
        records = [candidate("a", 1, "1", description=None)]
        scorer = FakeScorer()
        with self.assertRaisesRegex(ValueError, "missing its description"):
            rerank_candidates(records, scorer)
        self.assertEqual(scorer.calls, [])
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            rerank_candidates([candidate("a", 1, "1")], scorer, [])

    def test_trec_output_uses_original_ids(self):
        results = rerank_candidates([candidate("a", 1, "1")], FakeScorer())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run.txt.gz"
            write_reranked_run(results, path)
            with gzip.open(path, "rt") as stream:
                self.assertEqual(stream.read().split()[:5], ["1", "Q0", "a", "1", "1"])

    def test_chunk_coverage_overlap_and_query_are_preserved(self):
        chunks = pair_chunks(FakeTokenizer(), "99", "1 2 3 4 5 6 7 8 9", 8, 2)
        self.assertEqual([c["document_ids"] for c in chunks],
                         [[1,2,3,4], [3,4,5,6], [5,6,7,8], [7,8,9]])
        self.assertTrue(all(len(c["input_ids"]) <= 8 for c in chunks))
        self.assertTrue(all(c["input_ids"][1] == 99 for c in chunks))

    def test_description_not_silently_truncated_and_overlap_validated(self):
        with self.assertRaisesRegex(ValueError, "no document space"):
            pair_chunks(FakeTokenizer(), "1 2 3 4 5", "6", 8, 0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            pair_chunks(FakeTokenizer(), "1", "6", 8, 4)

    def test_best_late_chunk_determines_document_score(self):
        scorer = CrossEncoderScorer.__new__(CrossEncoderScorer)
        scorer.tokenizer, scorer.max_length, scorer.overlap = FakeTokenizer(), 8, 2
        scorer._score_features = lambda features: [-2, 1, 4, 0]
        result = scorer.score("99", "1 2 3 4 5 6 7 8 9")
        self.assertEqual(result, {"score": 4.0, "chunk_count": 4, "best_chunk_index": 2})
