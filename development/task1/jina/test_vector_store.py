import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vector_store import VectorWriter,search


class VectorStoreTest(unittest.TestCase):
    def test_persistent_index_and_best_chunk_aggregation(self):
        with tempfile.TemporaryDirectory() as d:
            w=VectorWriter(d)
            w.add([[1,0],[0,1]],[{'doc_id':'long'},{'doc_id':'long'}])
            w.add([[1,1]],[{'doc_id':'short'}]);w.close({'revision':'test'})
            result=search(d,[[1,0],[0,1]],batch_size=1)
            np.testing.assert_allclose(result['long'],[1,1])
            np.testing.assert_allclose(result['short'],[2**-.5]*2,rtol=1e-6)
            self.assertEqual(json.loads((Path(d)/'manifest.json').read_text())['rows'],3)
    def test_incomplete_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w=VectorWriter(d);w.add([[1,0]],[{'doc_id':'a'}]);w.close({})
            (Path(d)/'embeddings.f32').write_bytes(b'')
            with self.assertRaises(ValueError):search(d,[[1,0]])


if __name__=='__main__':unittest.main()
