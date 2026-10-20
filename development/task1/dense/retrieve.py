"""Exact dense-only chunk search over the complete corpus; no BM25 filtering."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

MODEL_ID = 'intfloat/multilingual-e5-small'
MODEL_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


def jsonl(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError('Invalid embedding')
    return vectors / norms


def search(queries, documents, model, batch_size=32, chunk_tokens=480, overlap=64):
    """Stream exact query/chunk cosines; max-pool by original document ID."""
    if batch_size < 1 or not 0 <= overlap < chunk_tokens <= 480:
        raise ValueError('Invalid batching or chunk settings')
    prompts = ['query: ' + q['query'].strip() for q in queries]
    if any(not q['query'].strip() for q in queries):
        raise ValueError('Empty query')
    if any(len(model.tokenizer.encode(p, add_special_tokens=True)) > 512 for p in prompts):
        raise ValueError('Query exceeds embedding model context')
    query_vectors = normalize(model.encode(prompts, batch_size=batch_size, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False))
    scores, seen, texts, owners = {}, set(), [], []
    counts = Counter()

    def flush():
        if not texts:
            return
        vectors = normalize(model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False))
        similarities = query_vectors @ vectors.T
        for i, owner in enumerate(owners):
            if owner not in scores:
                scores[owner] = similarities[:, i].copy()
            else:
                np.maximum(scores[owner], similarities[:, i], out=scores[owner])
        texts.clear()
        owners.clear()

    for document in documents:
        doc_id = str(document['doc_id'])
        if doc_id in seen:
            raise ValueError('Duplicate document ID')
        seen.add(doc_id)
        counts['documents_seen'] += 1
        text = document['text']
        if not isinstance(text, str) or not text.strip():
            counts['empty_documents_skipped'] += 1
            continue
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            counts['empty_documents_skipped'] += 1
            continue
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            passage = 'passage: ' + model.tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            if len(model.tokenizer.encode(passage, add_special_tokens=True)) > 512:
                raise ValueError('Decoded passage exceeds context; refusing silent truncation')
            texts.append(passage)
            owners.append(doc_id)
            counts['chunks_encoded'] += 1
            if len(texts) >= batch_size:
                flush()
            if end == len(tokens):
                break
            start = end - overlap
        if counts['documents_seen'] % 100 == 0:
            print(json.dumps(dict(counts)), flush=True)
    flush()
    if not scores:
        raise ValueError('No searchable documents')
    counts['documents_embedded'] = len(scores)
    return scores, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/models/multilingual-e5-small'))
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty')
    if not args.model.is_dir():
        parser.error('Local embedding model missing')
    args.output.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    model = SentenceTransformer(str(args.model), device='cpu', local_files_only=True)
    model.max_seq_length = 512
    queries = list(jsonl(args.dataset / 'queries.jsonl'))
    if not queries or len({str(q['qid']) for q in queries}) != len(queries):
        raise ValueError('Missing or duplicate query IDs')
    path = args.dataset / 'documents.jsonl.gz'
    if not path.exists():
        path = args.dataset / 'documents.jsonl'
    scores, counts = search(queries, jsonl(path), model, batch_size=args.batch_size)
    with gzip.open(args.output / 'run.txt.gz', 'wt') as stream:
        for i, query in enumerate(queries):
            ordered = sorted(scores, key=lambda doc_id: (-float(scores[doc_id][i]), doc_id))[:1000]
            for rank, doc_id in enumerate(ordered, 1):
                stream.write(f"{query['qid']} Q0 {doc_id} {rank} {scores[doc_id][i]:.9g} vera-dense-e5\n")
    summary = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'query_field': 'query',
               'query_prefix': 'query: ', 'document_prefix': 'passage: ', 'chunk_tokens': 480,
               'overlap': 64, 'aggregation': 'maximum chunk cosine per document',
               'search': 'exact full-corpus; no BM25 or cross-encoder', 'counts': counts,
               'queries': len(queries), 'seconds': time.monotonic()-started, 'cpu_threads': 4}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
