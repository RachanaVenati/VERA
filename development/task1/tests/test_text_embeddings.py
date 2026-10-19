"""Check long-document pooling without downloading or loading a model."""

import unittest
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from text_embeddings import DocumentEmbedder
from event_logging import EventTrace


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(int(text)))

    def decode(self, tokens):
        return str(tokens[0])


class TextEmbeddingsTest(unittest.TestCase):
    def test_later_chunks_are_included_and_weighted(self):
        calls = []
        def encode(chunks, **kwargs):
            calls.extend(chunks)
            return np.array([[1, 0], [0, 1]])
        embedder = DocumentEmbedder.__new__(DocumentEmbedder)
        embedder.model = SimpleNamespace(tokenizer=Tokenizer(), encode=encode)
        embedder.batch_size = 16
        vector = embedder.encode(["600"])[0]
        self.assertEqual(calls, ["query: 0", "query: 480"])
        expected = np.array([480, 120]) / np.linalg.norm([480, 120])
        np.testing.assert_allclose(vector, expected)

    def test_missing_local_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Local embedding model not found"):
            DocumentEmbedder("/model-path-that-does-not-exist")

    def test_embedding_failure_is_logged_with_correct_model(self):
        from text_embeddings import MODEL_ID
        embedder = DocumentEmbedder.__new__(DocumentEmbedder)
        def fail(*args, **kwargs):
            raise RuntimeError("embedding failed")
        embedder.model = SimpleNamespace(tokenizer=Tokenizer(), encode=fail)
        embedder.batch_size = 16
        embedder.records = [{"qid": "1", "doc_id": "doc", "text": "600"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gz"
            embedder.trace = EventTrace(path, "cross-encoder")
            root = embedder.trace.emit("1", "input")
            embedder.parents = {"1": root}
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                embedder.encode(["600"])
            embedder.trace.close()
            with gzip.open(path, "rt") as stream:
                event = [json.loads(line) for line in stream][-1]
        self.assertEqual(event["model"], MODEL_ID)
        self.assertEqual(event["event_type"], "model_call")
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["parent_event_id"], root)
