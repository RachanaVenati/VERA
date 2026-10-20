"""Field selection checks without Java, model weights, or dataset dependencies."""
import unittest
from types import SimpleNamespace

from query_fields import reranking_text, reranking_field, retrieval_text


class RerankingFallbackTest(unittest.TestCase):
    def test_description_takes_precedence(self):
        query = SimpleNamespace(query_id="1", query="travel", original_query={
            "description": " expense rules ", "narrative": "unused"})
        self.assertEqual(reranking_text(query), "expense rules")
        self.assertEqual(reranking_field(query), "description")
        self.assertEqual(retrieval_text(query), "travel")

    def test_unusable_descriptions_fall_back(self):
        for metadata in (None, {}, {"description": None}, {"description": ""},
                         {"description": " \t\n"}, {"description": 42}):
            with self.subTest(metadata=metadata):
                query = SimpleNamespace(query_id="1", query=" travel ", original_query=metadata)
                self.assertEqual(reranking_text(query), "travel")
                self.assertEqual(reranking_field(query), "query")
        self.assertEqual(reranking_text(SimpleNamespace(query_id="1", query="travel")), "travel")

    def test_missing_both_fails_without_using_narrative(self):
        query = SimpleNamespace(query_id="1", query=" ", original_query={"narrative": "unused"})
        with self.assertRaisesRegex(ValueError, "query field"):
            reranking_text(query)
