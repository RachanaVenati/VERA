import math
import unittest

import numpy as np

from semantic_duplicates import group_candidates


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.encoded = []

    def encode(self, texts):
        self.encoded = texts
        return np.array([self.vectors[text] for text in texts])


def candidate(doc_id, rank, text=None, qid="1", metadata=None):
    return {"qid": qid, "doc_id": doc_id, "bm25_rank": rank,
            "text": text or doc_id, "metadata": metadata or {}}


class SemanticDuplicatesTest(unittest.TestCase):
    def test_normalization_threshold_and_id_preservation(self):
        encoder = FakeEmbedder({"a": [10, 0], "b": [0.96, 0.28], "c": [0, 1]})
        groups = group_candidates([candidate("b", 2), candidate("a", 1), candidate("c", 3)], encoder)
        self.assertEqual([g["representative_id"] for g in groups], ["a", "c"])
        self.assertEqual([m["doc_id"] for m in groups[0]["members"]], ["a", "b"])
        self.assertAlmostEqual(groups[0]["members"][1]["cosine_to_representative"], 0.96)

    def test_no_transitive_chaining(self):
        angle = math.radians(15)
        encoder = FakeEmbedder({"a": [1, 0], "b": [math.cos(angle), math.sin(angle)],
                                "c": [math.cos(2*angle), math.sin(2*angle)]})
        groups = group_candidates([candidate("a", 1), candidate("b", 2), candidate("c", 3)], encoder)
        self.assertEqual(len(groups), 2)

    def test_groups_are_per_query_and_embedding_text_is_reused(self):
        encoder = FakeEmbedder({"same": [1, 0]})
        records = [candidate("a", 1, "same"), candidate("b", 2, "same"),
                   candidate("a", 1, "same", qid="2")]
        groups = group_candidates(records, encoder)
        self.assertEqual(len(groups), 2)
        self.assertEqual(encoder.encoded, ["same"])

    def test_different_legal_decisions_never_merge_or_select_newest(self):
        encoder = FakeEmbedder({"same": [1, 0]})
        records = [candidate("old", 1, "same", metadata={"ecli": "one", "decision_date": "2020"}),
                   candidate("new", 2, "same", metadata={"ecli": "two", "decision_date": "2024"})]
        self.assertEqual(len(group_candidates(records, encoder)), 2)

    def test_matching_legal_identity_can_group(self):
        encoder = FakeEmbedder({"same": [1, 0]})
        records = [candidate("a", 1, "same", metadata={"ecli": "one"}),
                   candidate("b", 2, "same", metadata={"ecli": "one"})]
        self.assertEqual(len(group_candidates(records, encoder)), 1)

    def test_invalid_vectors_and_threshold_fail(self):
        records = [candidate("a", 1)]
        for vector in ([0, 0], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                group_candidates(records, FakeEmbedder({"a": vector}))
        with self.assertRaises(ValueError):
            group_candidates([], None, 1.1)
        self.assertEqual(group_candidates([], None), [])
