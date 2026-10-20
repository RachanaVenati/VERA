"""Append-only local vector files plus exact, memory-mapped document search."""
import json
from pathlib import Path
import numpy as np


class VectorWriter:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            raise ValueError('Vector directory must be empty')
        self.path.mkdir(parents=True, exist_ok=True)
        self.vectors = open(self.path/'embeddings.f32', 'wb')
        self.chunks = open(self.path/'chunks.jsonl', 'w')
        self.count, self.dimension = 0, None

    def add(self, vectors, metadata):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(metadata) or not np.isfinite(vectors).all():
            raise ValueError('Invalid vector batch')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Zero vector')
        if self.dimension is not None and vectors.shape[1] != self.dimension:
            raise ValueError('Embedding dimensions changed')
        self.dimension = vectors.shape[1]
        (vectors/norms).astype('<f4').tofile(self.vectors)
        for item in metadata:
            self.chunks.write(json.dumps({**item, 'row':self.count})+'\n')
            self.count += 1

    def abort(self):
        self.vectors.close()
        self.chunks.close()

    def close(self, manifest):
        self.vectors.close(); self.chunks.close()
        if not self.count:
            raise ValueError('No embeddings written')
        (self.path/'manifest.json').write_text(json.dumps({**manifest, 'rows':self.count,
            'dimension':self.dimension, 'dtype':'<f4', 'format_version':1, 'normalized':True}, indent=2)+'\n')


def search(path, queries, batch_size=4096):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    rows, dimension = manifest['rows'], manifest['dimension']
    if (path/'embeddings.f32').stat().st_size != rows*dimension*4:
        raise ValueError('Vector file size mismatch')
    vectors = np.memmap(path/'embeddings.f32', dtype='<f4', mode='r', shape=(rows,dimension))
    queries = np.asarray(queries, dtype=np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if not np.isfinite(queries).all() or np.any(norms==0):
        raise ValueError('Invalid query vectors')
    if queries.ndim != 2 or queries.shape[1] != dimension:
        raise ValueError('Query embedding dimension mismatch')
    queries = queries/norms
    scores = {}
    with open(path/'chunks.jsonl') as stream:
        for start in range(0, rows, batch_size):
            end=min(start+batch_size,rows)
            similarities=queries @ vectors[start:end].T
            for i in range(end-start):
                line=next(stream, None)
                if line is None:
                    raise ValueError('Missing chunk mapping rows')
                item=json.loads(line)
                if item['row'] != start+i:
                    raise ValueError('Chunk mapping row mismatch')
                owner=item['doc_id']
                if owner not in scores:
                    scores[owner]=similarities[:,i].copy()
                else:
                    np.maximum(scores[owner], similarities[:,i], out=scores[owner])
        if stream.read().strip():
            raise ValueError('Extra chunk mapping rows')
    return scores
