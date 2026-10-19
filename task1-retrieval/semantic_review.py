"""Audit semantic candidate pairs and calibrate a threshold from human labels.

Similarity proposes pairs; it never proves equivalence or authorizes deletion.
All comparisons are within a query. Pair labels describe interchangeable
information, not merely common subject matter.
"""

import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from semantic_duplicates import _same_legal_record, group_candidates


def audit_candidates(records, embedder, threshold=0.95, review_floor=0.90):
    """Encode each distinct text once, return groups and reviewable pairs.

    Every above-floor pair is exported, including legally blocked pairs and
    pairs not assigned to the same greedy representative group. The review
    floor bounds the range in which thresholds can subsequently be calibrated.
    """
    if not 0 < review_floor <= threshold <= 1:
        raise ValueError("Require 0 < review floor <= threshold <= 1.")
    if not records:
        return [], []
    texts = list(dict.fromkeys(r["text"] for r in records))
    vectors = np.asarray(embedder.encode(texts), dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding vectors.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero embedding vector.")
    vectors = vectors / norms[:, None]
    lookup = dict(zip(texts, vectors))

    class CachedEmbedder:
        def encode(self, requested):
            return np.asarray([lookup[t] for t in requested])

    groups = group_candidates(records, CachedEmbedder(), threshold)
    by_query = {}
    for record in sorted(records, key=lambda r: (r["qid"], r["bm25_rank"], r["doc_id"])):
        by_query.setdefault(record["qid"], []).append(record)
    pairs = []
    for qid, candidates in by_query.items():
        # Hold out complete queries, avoiding the same query on both sides.
        split = "validation" if int(hashlib.sha256(str(qid).encode()).hexdigest(), 16) % 5 == 0 else "calibration"
        for left, right in combinations(candidates, 2):
            cosine = float(np.clip(lookup[left["text"]] @ lookup[right["text"]], -1, 1))
            if cosine < review_floor:
                continue
            allowed = _same_legal_record(left, right)
            pairs.append({
                "qid": qid, "left_id": left["doc_id"], "right_id": right["doc_id"],
                "left_bm25_rank": left["bm25_rank"], "right_bm25_rank": right["bm25_rank"],
                "cosine": cosine, "review_floor": review_floor, "split": split,
                "legal_identity_allowed": allowed,
                "proposed_grouping": allowed and cosine >= threshold,
                "description": left.get("description", ""),
                "left_metadata": left.get("metadata", {}), "right_metadata": right.get("metadata", {}),
                "left_text": left["text"], "right_text": right["text"],
                "label": "", "notes": "",
            })
    return groups, pairs


def write_review(pairs, path):
    """Full text is intentional: short excerpts can hide crucial differences."""
    fields = ["qid", "left_id", "right_id", "left_bm25_rank", "right_bm25_rank",
              "cosine", "review_floor", "split", "legal_identity_allowed", "proposed_grouping",
              "description", "left_metadata", "right_metadata", "left_text", "right_text", "label", "notes"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = dict(pair)
            for key in ("left_metadata", "right_metadata"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def calibrate(rows, thresholds, target_precision=0.98, min_predictions=5):
    """Select on calibration queries only; report held-out performance separately.

    Precision is conditional on reviewed above-floor pairs, not corpus-wide.
    No threshold is recommended without labeled held-out positive/negative pairs.
    """
    if not 0 < target_precision <= 1 or min_predictions < 1:
        raise ValueError("Invalid precision target or minimum prediction count.")
    labeled, seen = [], set()
    for row in rows:
        label = row["label"].strip().lower()
        if label not in ("", "uncertain", "duplicate", "distinct"):
            raise ValueError(f"Unknown label: {label}")
        key = (str(row["qid"]), *sorted((str(row["left_id"]), str(row["right_id"]))))
        if key in seen:
            raise ValueError(f"Repeated review pair: {key}")
        seen.add(key)
        if label in ("", "uncertain"):
            continue
        split = row["split"]
        if split not in ("calibration", "validation"):
            raise ValueError("Split must be calibration or validation.")
        score, floor = float(row["cosine"]), float(row["review_floor"])
        if not np.isfinite(score) or not -1 <= score <= 1 or not 0 < floor <= 1:
            raise ValueError("Invalid score or review floor.")
        allowed = str(row["legal_identity_allowed"]).lower()
        if allowed not in ("true", "false"):
            raise ValueError("Invalid legal identity flag.")
        labeled.append({"split": split, "positive": label == "duplicate", "score": score,
                        "floor": floor, "allowed": allowed == "true", "qid": str(row["qid"])})
    if {r["qid"] for r in labeled if r["split"] == "calibration"} & {r["qid"] for r in labeled if r["split"] == "validation"}:
        raise ValueError("A query cannot appear in both splits.")
    thresholds = sorted(set(thresholds))
    if not thresholds or any(not np.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("Thresholds must be in (0, 1].")
    if labeled and min(thresholds) < max(r["floor"] for r in labeled):
        raise ValueError("Cannot calibrate below the exported review floor.")

    def metrics(split, threshold):
        subset = [r for r in labeled if r["split"] == split]
        predicted = [r for r in subset if r["allowed"] and r["score"] >= threshold]
        tp = sum(r["positive"] for r in predicted)
        positives = sum(r["positive"] for r in subset)
        return {"labeled": len(subset), "positives": positives, "predicted_duplicates": len(predicted),
                "true_positives": tp, "false_positives": len(predicted) - tp,
                "precision": tp / len(predicted) if predicted else None,
                "recall_within_reviewed_pairs": tp / positives if positives else None}

    grid = [{"threshold": t, "calibration": metrics("calibration", t),
             "validation": metrics("validation", t)} for t in thresholds]
    eligible = [r for r in grid if r["calibration"]["predicted_duplicates"] >= min_predictions
                and r["calibration"]["precision"] >= target_precision]
    selected = min(eligible, key=lambda r: (-(r["calibration"]["recall_within_reviewed_pairs"] or 0), r["threshold"])) if eligible else None
    validation = selected["validation"] if selected else None
    supported = bool(validation and 0 < validation["positives"] < validation["labeled"]
                     and validation["predicted_duplicates"] >= min_predictions
                     and validation["precision"] >= target_precision)
    return {"grid": grid, "selected_on_calibration": selected["threshold"] if selected else None,
            "recommended_for_further_testing": selected["threshold"] if selected and supported else None,
            "target_precision": target_precision, "min_predictions": min_predictions,
            "policy": "No deletion. Metrics apply only to reviewed above-floor pairs; validate retrieval separately."}
