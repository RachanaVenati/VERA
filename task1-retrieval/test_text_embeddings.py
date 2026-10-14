"""Check long-document pooling without downloading or loading a model."""

import unittest
from types import SimpleNamespace

import numpy as np

from text_embeddings import DocumentEmbedder


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
