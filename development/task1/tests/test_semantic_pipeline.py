"""Exercise the real CLI/audit/merge with small deterministic model doubles.

BM25/JVM and model weights are replaced; field handling, candidate preparation,
semantic audit artifacts, reranking, tracing, and ranking merge are real code.
"""

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd

from query_fields import retrieval_text
from text_embeddings import DocumentEmbedder, MODEL_ID as EMBEDDING_MODEL


def load_pipeline():
    baseline = ModuleType("baseline")
    baseline.create_index = lambda *args, **kwargs: None
    baseline.detect_query_language = lambda data: "en"
    baseline.retrieve = lambda *args, **kwargs: None
    integrations = ModuleType("tira.third_party_integrations")
    integrations.ir_datasets = SimpleNamespace(load=lambda path: None)
    path = Path(__file__).resolve().parents[3] / "task1-retrieval/submission.py"
    spec = importlib.util.spec_from_file_location("semantic_test_submission", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"baseline": baseline, "tira.third_party_integrations": integrations}):
        spec.loader.exec_module(module)
    return module


class SemanticPipelineTest(unittest.TestCase):
    def test_query_for_bm25_description_for_reranking_audit_preserves_ids(self):
        pipeline = load_pipeline()
        query = SimpleNamespace(query_id="1", query="travel", original_query={
            "language": "en", "description": "eligibility for mobility funding", "narrative": "unused narrative"})
        documents = [SimpleNamespace(doc_id="one", default_text=lambda: "600"),
                     SimpleNamespace(doc_id="two", default_text=lambda: "700")]
        data = SimpleNamespace(queries_iter=lambda: iter([query]), docs_iter=lambda: iter(documents))
        stages, calls = [], []

        def index(data, language, output, **kwargs):
            for name in ("index-ir-metadata.yml", "retrieval-ir-metadata.yml"):
                (output / name).write_text("{}")
            return SimpleNamespace(close=lambda: None)

        def retrieve(data, *args):
            stages.append("bm25")
            self.assertEqual([retrieval_text(q) for q in data.queries_iter()], ["travel"])
            return pd.DataFrame([["1", "one", 0, 10], ["1", "two", 1, 9]],
                                columns=["qid", "docno", "rank", "score"])

        def embedding_model(path, batch_size):
            stages.append("embedding")
            model = DocumentEmbedder.__new__(DocumentEmbedder)
            model.batch_size = batch_size
            model.model = SimpleNamespace(tokenizer=SimpleNamespace(
                encode=lambda text, **kwargs: list(range(int(text))), decode=lambda ids: str(ids[0])),
                encode=lambda chunks, **kwargs: [[1.0, 0.0] for _ in chunks])
            return model

        class Scorer:
            def __init__(self, **kwargs):
                pass

            def score(self, description, text):
                stages.append("rerank")
                calls.append((description, text))
                return {"score": float(text), "chunk_count": 1, "best_chunk_index": 0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            model_path.mkdir()
            with patch.object(pipeline.ir_datasets, "load", return_value=data), \
                    patch.object(pipeline, "create_index", side_effect=index), \
                    patch.object(pipeline, "retrieve", side_effect=retrieve), \
                    patch.object(pipeline, "CrossEncoderScorer", Scorer), \
                    patch("text_embeddings.DocumentEmbedder", side_effect=embedding_model):
                result = CliRunner().invoke(pipeline.main, ["--dataset", str(root), "--output", str(root / "out"),
                    "--semantic-mode", "audit", "--semantic-model-path", str(model_path)])
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["retrieval_field"], "query")
            self.assertEqual(summary["reranking_field"], "description")
            self.assertEqual(summary["semantic_audit"]["multi_member_groups"], 1)
            with gzip.open(root / "out/run.txt.gz", "rt") as stream:
                self.assertEqual([line.split()[2] for line in stream], ["two", "one"])
            with gzip.open(root / "out/run-trace.jsonl.log.gz", "rt") as stream:
                events = [json.loads(line) for line in stream]
            embeddings = [e for e in events if e["event_type"] == "model_call"]
            self.assertEqual(len(embeddings), 2)
            self.assertTrue(all(e["model"] == EMBEDDING_MODEL for e in embeddings))
            self.assertEqual(events[-1]["event_type"], "decision")
            self.assertTrue((root / "out/semantic-review-pairs.csv").is_file())
            with patch.object(pipeline.ir_datasets, "load", return_value=data), \
                    patch.object(pipeline, "create_index", side_effect=index), \
                    patch.object(pipeline, "retrieve", side_effect=retrieve), \
                    patch.object(pipeline, "CrossEncoderScorer", Scorer), \
                    patch("text_embeddings.DocumentEmbedder") as unused_embedder:
                previous_stages, previous_calls = len(stages), len(calls)
                off = CliRunner().invoke(pipeline.main, ["--dataset", str(root), "--output", str(root / "off")])
                self.assertEqual(off.exit_code, 0, off.output + repr(off.exception))
                unused_embedder.assert_not_called()
                self.assertFalse((root / "off/semantic-review-pairs.csv").exists())
                self.assertEqual(json.loads((root / "off/summary.json").read_text())["semantic_audit"], {"mode": "off"})
                del stages[previous_stages:]
                del calls[previous_calls:]
        self.assertEqual(stages, ["bm25", "embedding", "rerank", "rerank"])
        self.assertEqual(calls, [(query.original_query["description"], "600"),
                                 (query.original_query["description"], "700")])

    def test_missing_or_blank_description_reaches_cross_encoder_as_query(self):
        pipeline = load_pipeline()
        queries = [SimpleNamespace(query_id=str(i), query="travel", original_query=metadata)
                   for i, metadata in enumerate([
                       {}, {"description": None}, {"description": "  "},
                       {"description": " expense rules "},
                   ])]
        data = SimpleNamespace(queries_iter=lambda: iter(queries), docs_iter=lambda: iter([
            SimpleNamespace(doc_id="one", default_text=lambda: "travel reimbursement")]))
        calls = []

        def index(data, language, output, **kwargs):
            for name in ("index-ir-metadata.yml", "retrieval-ir-metadata.yml"):
                (output / name).write_text("{}")
            return SimpleNamespace(close=lambda: None)

        def retrieve(data, *args):
            self.assertEqual([retrieval_text(q) for q in data.queries_iter()], ["travel"] * 4)
            return pd.DataFrame([[q.query_id, "one", 0, 10] for q in queries],
                                columns=["qid", "docno", "rank", "score"])

        class Scorer:
            def __init__(self, **kwargs):
                pass

            def score(self, description, text):
                calls.append(description)
                return {"score": 1.0, "chunk_count": 1, "best_chunk_index": 0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(pipeline.ir_datasets, "load", return_value=data), \
                    patch.object(pipeline, "create_index", side_effect=index), \
                    patch.object(pipeline, "retrieve", side_effect=retrieve), \
                    patch.object(pipeline, "CrossEncoderScorer", Scorer):
                result = CliRunner().invoke(pipeline.main, ["--dataset", str(root), "--output", str(root / "out")])
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            # Identical effective query/document pairs share a score across questions.
            self.assertEqual(calls, ["travel", "expense rules"])
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["reranking_fields_by_query"],
                             {"0": "query", "1": "query", "2": "query", "3": "description"})
            with gzip.open(root / "out/run.txt.gz", "rt") as stream:
                self.assertEqual(len(list(stream)), 4)

    def test_missing_audit_model_fails_before_indexing(self):
        pipeline = load_pipeline()
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "create_index") as index:
            result = CliRunner().invoke(pipeline.main, ["--dataset", directory, "--output", directory + "/out",
                "--semantic-mode", "audit", "--semantic-model-path", directory + "/absent"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Semantic model is missing", result.output)
            index.assert_not_called()

    def test_group_ranking_requires_audit_before_indexing(self):
        pipeline = load_pipeline()
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "create_index") as index:
            for strategy in ("group-winners", "mmr"):
                result = CliRunner().invoke(pipeline.main, ["--dataset", directory,
                    "--output", directory + "/out", "--ranking-strategy", strategy])
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("Group ranking requires", result.output)
            index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
