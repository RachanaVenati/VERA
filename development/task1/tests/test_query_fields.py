"""Verify field separation against real dataset-loader objects."""

import unittest

from query_fields import reranking_text, retrieval_text
from test_baseline import persisted_dataset
from baseline import create_index, retrieve


class QueryFieldsTest(unittest.TestCase):
    def test_nested_description_is_separate_from_bm25_query(self):
        queries = [{
            "qid": "travel", "query": "travel",
            "original_query": {
                "language": "en", "description": "laboratory microscope",
                "narrative": "library books",
            },
        }]
        documents = [
            {"doc_id": "travel-doc", "text": "travel reimbursement"},
            {"doc_id": "description-doc", "text": "laboratory microscope"},
            {"doc_id": "narrative-doc", "text": "library books"},
        ]
        with persisted_dataset(queries, documents) as dataset:
            query = next(dataset.queries_iter())
            self.assertEqual(retrieval_text(query), "travel")
            self.assertEqual(reranking_text(query), "laboratory microscope")
            run = retrieve(dataset, create_index(dataset, "en"), "en")
            self.assertEqual(set(run.docno), {"travel-doc"})

    def test_description_outside_official_metadata_uses_query_fallback(self):
        with persisted_dataset(
            [{"qid": "1", "query": "travel", "description": "expense rules"}],
            [{"doc_id": "1", "text": "travel"}],
        ) as dataset:
            self.assertEqual(reranking_text(next(dataset.queries_iter())), "travel")

    def test_missing_description_does_not_break_bm25(self):
        with persisted_dataset(
            [{"qid": "1", "query": "travel", "original_query": {"language": "en"}}],
            [{"doc_id": "1", "text": "travel"}],
        ) as dataset:
            query = next(dataset.queries_iter())
            self.assertEqual(retrieval_text(query), "travel")
            self.assertEqual(reranking_text(query), "travel")

    def test_missing_query_is_not_replaced_by_description(self):
        from types import SimpleNamespace
        query = SimpleNamespace(query_id="1", query=" ", description="travel")
        with self.assertRaisesRegex(ValueError, "query field"):
            retrieval_text(query)
