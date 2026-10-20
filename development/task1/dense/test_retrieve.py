import unittest
from types import SimpleNamespace
import numpy as np
from retrieve import search, normalize


class Model:
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split(),
                                decode=lambda ids, **kwargs: ' '.join(ids))
    def encode(self, texts, **kwargs):
        assert all(t.startswith(('query: ', 'passage: ')) for t in texts)
        return np.array([[text.count('alpha'), text.count('beta')] for text in texts], dtype=float)


class DenseSearchTest(unittest.TestCase):
    def test_full_corpus_chunk_max_pool_and_query_prefix(self):
        queries = [{'qid':'1','query':'alpha'}, {'qid':'2','query':'beta'}]
        documents = [{'doc_id':'long','text':'beta beta alpha alpha'},
                     {'doc_id':'short','text':'alpha beta'}, {'doc_id':'empty','text':''}]
        scores, counts = search(queries, documents, Model(), batch_size=1, chunk_tokens=2, overlap=0)
        np.testing.assert_allclose(scores['long'], [1,1])
        np.testing.assert_allclose(scores['short'], [2**-.5,2**-.5], rtol=1e-6)
        self.assertEqual(counts['chunks_encoded'], 3)
        self.assertEqual(counts['empty_documents_skipped'], 1)
        self.assertEqual(counts['documents_seen'], 3)

    def test_invalid_embedding_and_duplicate_id_fail(self):
        with self.assertRaises(ValueError):
            normalize([[0,0]])
        doc = {'doc_id':'a','text':'alpha'}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            search([{'qid':'1','query':'alpha'}], [doc, doc], Model())


if __name__ == '__main__':
    unittest.main()
