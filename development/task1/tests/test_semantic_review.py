import math
import csv
from pathlib import Path
import tempfile
import unittest

from semantic_review import audit_candidates, calibrate, write_review


class Encoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [self.vectors[t] for t in texts]


def record(doc, rank, metadata=None, qid="1"):
    return {"qid": qid, "doc_id": doc, "text": doc, "bm25_rank": rank,
            "metadata": metadata or {}, "description": "Eligibility?"}


def review(left, score, positive, split="calibration", allowed=True, qid=None):
    return {"qid": qid or split, "left_id": left, "right_id": "other", "cosine": score,
            "review_floor": 0.90, "legal_identity_allowed": allowed, "split": split,
            "label": "duplicate" if positive else "distinct"}


class SemanticReviewTest(unittest.TestCase):
    def test_blocked_legal_pair_is_visible_without_merging(self):
        encoder = Encoder({"a": [1, 0], "b": [1, 0]})
        groups, pairs = audit_candidates([record("a", 1, {"ecli": "one"}),
                                          record("b", 2, {"ecli": "two"})], encoder)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["legal_identity_allowed"])
        self.assertFalse(pairs[0]["proposed_grouping"])
        self.assertEqual(encoder.calls, 1)

    def test_related_pair_below_threshold_is_available_for_review(self):
        encoder = Encoder({"a": [1, 0], "b": [0.94, math.sqrt(1 - 0.94**2)]})
        groups, pairs = audit_candidates([record("a", 1), record("b", 2)], encoder)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["proposed_grouping"])
        self.assertEqual(pairs[0]["left_text"], "a")

    def test_no_cross_query_pairs(self):
        encoder = Encoder({"a": [1, 0]})
        groups, pairs = audit_candidates([record("a", 1), record("a", 1, qid="2")], encoder)
        self.assertEqual(len(groups), 2)
        self.assertEqual(pairs, [])

    def test_csv_round_trip_preserves_full_text_and_supports_calibration(self):
        encoder = Encoder({'a,\n"full text"': [1, 0], "b": [1, 0]})
        _, pairs = audit_candidates([record('a,\n"full text"', 1), record("b", 2)], encoder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_review(pairs, path)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["left_text"], 'a,\n"full text"')
        rows[0]["label"] = "duplicate"
        self.assertEqual(calibrate(rows, [0.95], min_predictions=1)["grid"][0]["calibration"]["labeled"]
                         + calibrate(rows, [0.95], min_predictions=1)["grid"][0]["validation"]["labeled"], 1)

    def test_no_recommendation_without_held_out_support(self):
        report = calibrate([review("a", 0.99, True)], [0.95, 0.99], min_predictions=1)
        self.assertEqual(report["selected_on_calibration"], 0.95)
        self.assertIsNone(report["recommended_for_further_testing"])

    def test_selection_uses_calibration_and_reports_heldout_false_positive(self):
        rows = [review("a", 0.99, True), review("b", 0.96, False),
                review("c", 0.99, False, "validation"), review("d", 0.98, True, "validation")]
        result = calibrate(rows, [0.95, 0.98], min_predictions=1)
        self.assertEqual(result["selected_on_calibration"], 0.98)
        self.assertEqual(result["grid"][1]["validation"]["false_positives"], 1)
        self.assertIsNone(result["recommended_for_further_testing"])

    def test_recommendation_requires_validation_precision(self):
        rows = [review("a", 0.99, True), review("b", 0.93, False),
                review("c", 0.99, True, "validation"), review("d", 0.93, False, "validation")]
        self.assertEqual(calibrate(rows, [0.95], min_predictions=1)["recommended_for_further_testing"], 0.95)

    def test_review_scope_and_query_leakage_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate([review("a", 0.99, True)], [0.85])
        with self.assertRaises(ValueError):
            calibrate([review("a", 0.99, True, qid="same"),
                       review("b", 0.99, True, "validation", qid="same")], [0.95])

    def test_legal_guard_and_uncertain_labels(self):
        rows = [review("a", 1.0, True, allowed=False), review("b", 0.99, False)]
        rows[1]["label"] = "uncertain"
        result = calibrate(rows, [0.95], min_predictions=1)
        self.assertEqual(result["grid"][0]["calibration"]["predicted_duplicates"], 0)
        self.assertIsNone(result["selected_on_calibration"])


if __name__ == "__main__":
    unittest.main()
